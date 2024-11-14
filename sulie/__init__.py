import logging
import io
import math
import numpy as np
import os
import pandas as pd
import requests
import uuid
from pandas.tseries.frequencies import to_offset
from typing import Optional, List, Dict, Any, Literal, Union
from tqdm import tqdm

__version__ = "1.0.4"

logger = logging.getLogger("sulie")

_DEFAULT_API_URL = "https://api.sulie.co"


class SulieError(Exception):
    
    def __init__(self, code: int, detail: Optional[str]):
        self.code = code
        self.detail = detail

    def __repr__(self) -> str:
        return f"{self.detail} {self.code}"


# Custom API error codes
_DATASET_NOT_FOUND = 6500
_UPLOAD_IN_PROGRESS = 6505
_MODEL_NOT_FOUND = 6601
_UPGRADE_PLAN = 6754

# Supported dataset upload types
_UploadModes = Literal["append", "overwrite"]


class Sulie:

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Sulie instance."""
        self.api_url = os.environ.get("SULIE_API_URL") or _DEFAULT_API_URL

        self.api_key = api_key or os.environ.get("SULIE_API_KEY")
        if self.api_key is None:
            raise ValueError("No API key provided")
        else:
            self._caller = self._validate_api_key()["organization_id"]

    def list_datasets(self) -> List['Dataset']:
        """Query and return available datasets for a Sulie organization.
        
        Returns:
            List: Dataset names.
        """
        r = self._api_request("/datasets", "get")
        r.raise_for_status()

        datasets = []
        for kwargs in r.json():
            dataset = Dataset(self, **kwargs)
            datasets.append(dataset)

        return datasets
     
    def get_dataset(self, name: str) -> 'Dataset':
        """Query the database and return a dataset.
        
        Args:
            name (str): Name of the dataset.
        
        Returns:
            Dataset: Dataset.
        """
        dataset = Dataset.get(self, name)
        return dataset
    
    def upload_dataset(
            self, 
            df: pd.DataFrame,
            name: str, 
            mode: _UploadModes = "append", 
            storage: Literal["empheral", "persisted"] = "persisted",
            **kwargs
        ) -> 'Dataset':
        """Upload a dataset to the remote object store.
        
        Args:
            df (pd.DataFrame): Pandas data frame.
            name (str): Name of the dataset.
            mode (str): Upload mode.
            storage (str): Dataset storage class, default persisted.
            kwargs (dict): Optional keyword arguments, description.

        Returns:
            Dataset: Dataset.
        """
        try:
            dataset: Dataset = Dataset.get(self, name)
        except SulieError as e:
            if e.code != _DATASET_NOT_FOUND:
                raise
            description = kwargs.get("description")
            dataset = Dataset.create(self, name, description, mode=storage)
        finally:
            dataset.upload(df, mode)
            return dataset
        
    def list_custom_models(self) -> List["Model"]:
        """Query API for list of custom or fine-tuned models.
        
        Returns:
            List: List of models.
        """
        return Model.list(self)
        
    def get_model(self, model_name: str) -> "Model":
        """Load a fine-tuned model variant.
        
        Args:
            model_name (str): Name of the model.

        Returns:
            Model: Model instance.
        """
        return Model.get(self, model_name)

    def forecast(
            self, 
            dataset: Union["Dataset", pd.DataFrame],
            target: str = None,
            group_by: str = None,
            date: str = None,
            aggr: str = None,
            frequency: Literal["H", "D", "W", "M", "Y"] = "D",
            horizon: int = 30,
            n_rows: int = None,
            model: "Model" = None,
            **kwargs
        ) -> List[List[float]]:
        """Forecast using the time series foundation model.
        
        Args:
            dataset (Dataset): Dataset or pd.Dataframe.
            target (str): Name of the target value to forecast.
            group_by (str): Column to group the records on, optional.
            date (str): Name of the date (and time) column, optional.
            aggr (str): Column aggregation, required if `date` is set.
            frequency (str): Date aggregation frequency, optional.
            horizon (int): Prediction horizon length.
            n_rows (int): Number of rows to trim from tail.
            model (Model): Default or custom model for inference, optional.
            kwargs (dict): Optional inference keyword arguments.

        Returns:
            List: List of predictions.
        """
        model: Model = model or Model(self)
        r = model.forecast(
            dataset=dataset, 
            target=target, 
            group_by=group_by, 
            date=date, 
            aggr=aggr, 
            frequency=frequency, 
            horizon=horizon, 
            n_rows=n_rows, 
            **kwargs
        )
        return r
    
    def embed(self, arr: np.ndarray, model: "Model" = None) -> List[Any]:
        """Extract encoder embeddings.
        
        Args:
            arr (np.ndarray): List of values, 1D or 2D array.
            model (Model): Default or custom model for embeddings, optional.

        Returns:
            List: Encoder embeddings.
        """
        model: Model = model or Model(self)
        r = model.embed(arr)
        return r

    def fine_tune(
            self,
            dataset: Union["Dataset", pd.DataFrame],
            target: str,
            description: Optional[str] = None
        ) -> "FineTuneJob":
        """Run a model fine tuning job.
        
        Args:
            dataset (Dataset): Dataset or pd.Dataframe.
            target (str): Name of the target value to optimise for.
            description (str): Description of the fine-tune job.

        Returns:
            FineTuneJob: Fine-tuning job.
        """
        if target not in dataset.columns:
            raise KeyError(f"Target column {target} not found in dataset")

        min_dataset_size = 1000
        if len(dataset[target]) < min_dataset_size:
            raise ValueError(f"Requires at least {min_dataset_size} samples")

        # Create an empheral training dataset
        dataset_name = uuid.uuid4().__str__()
        if isinstance(dataset, Dataset):
            dataset_name = f"{dataset_name}-{dataset.name}"

        train_dataset = self.upload_dataset(
            df=dataset, 
            name=dataset_name, 
            mode="overwrite",
            storage="empheral"
        )

        # Start the fine-tune job
        job = FineTuneJob.fit(self, train_dataset.id, target, description)
        return job
    
    def list_fine_tuning_jobs(self) -> List["FineTuneJob"]:
        """Query API for list of fine tuning jobs.
        
        Returns:
            List: List of fine tuning jobs.
        """
        return FineTuneJob.list(self)

    def _validate_api_key(self):
        """Validate API key.
        
        Raises:
            ValueError: If API key is invalid or expired.
        """
        r = self._api_request("/keys", "put", headers={"Api-Key": self.api_key})
        r.raise_for_status()
        return r.json()
    
    def _api_request(
            self,
            endpoint: str, 
            method: Literal["get", "post", "put", "delete"], 
            secure: bool = True,
            **kwargs
        ) -> requests.Response:
        """Invoke an API endpoint.
        
        Args:
            endpoint (str): URL endpoint.
            method (str): HTTP method.
            secure (bool): Whether to run the endpoint using authentication.
            kwargs (dict): Request keyword arguments.
        """
        kwargs["url"] = self.api_url + endpoint

        if secure is True:
            headers = kwargs.get("headers", {})
            default = self._default_headers()
            kwargs["headers"] = {**headers, **default}

        f = getattr(requests, method.lower())
        r: requests.Response = f(**kwargs)
        return r

    def _default_headers(self):
        """Return the default request headers.
        
        Returns:
            Dict: Headers.
        """
        return {"Api-Key": self.api_key}

    def _handle_error(response: requests.Response, action: str):
        """Handle errors from API responses.
        
        Args:
            response (requests.Response): API endpoint response.
            action (str): Name of the action that was performed.
        
        Raises:
            Exception: An exception with action message and status code.
        """
        if response.status_code == 403:
            message = f'Authentication failed, invalid API key for {action}.'
        else:
            message = f'Unexpected error during {action}.'

        data = response.json()
        error_msg = data.get('error', 'No error details provided')

        message += error_msg

        raise requests.exceptions.HTTPError(message, response=response)


class Dataset(pd.DataFrame):
    
    def __init__(self, client: Sulie, **kwargs):
        super().__init__()
        self._client = client
        self.name = kwargs["name"]
        self.description = kwargs["description"]
        self.id = kwargs["dataset_id"]

    def __repr__(self):
        return f"{self.id} {self.name} {self.description}"

    @classmethod
    def _from_dict(cls, client: Sulie, **kwargs):
        """Initialize a Dataset instance from keyword arguments.
        
        Args:
            client (Sulie): API client.
            kwargs (dict): Keyword arguments.
        """
        dataset = cls(client, **kwargs)
        return dataset

    @staticmethod
    def _validate(df: pd.DataFrame):
        """Validate a pandas DataFrame.
        
        Args:
            df (pd.DataFrame): Data frame.

        Raises:
            AssertError: If any of the data types do not meet expectations.
        """
        func = pd.api.types.is_datetime64_any_dtype
        detail = "Dataset requires a datetime dimension, use `pd.to_datetime`"
        assert any([func(df[col]) for col in df.columns]), detail
    
    @staticmethod
    def get(client: Sulie, name: str) -> Dict[str, Any]:
        """Query API for dataset metadata.
        
        Args:
            name (str): Name of the dataset.

        Returns:
            dict: Dataset metadata.
        """
        params = {
            "name": name
        }
        r = client._api_request("/datasets", method="get", params=params)
        if r.status_code == 404:
            detail = "Dataset %s does not exist" % name
            raise SulieError(_DATASET_NOT_FOUND, detail)
        
        return Dataset(client, **r.json()[0])

    @staticmethod
    def create(
            client: Sulie, 
            name: str, 
            description: str, 
            **kwargs
        ) -> Dict[str, Any]:
        """Create a dataset.
        
        Args:
            name (str): Name of the dataset.
            description (str): Dataset freeform description.
            kwargs (dict): Optional keyword arguments.

        Returns:
            Dataset: Dataset.
        """
        body = {
            "name": name,
            "description": description,
            "organization_id": client._caller,
            "source": "pypi",
            "type": "custom",
            "mode": kwargs.get("mode", "persisted")
        }
        r = client._api_request("/datasets", "post", json=body)
        r.raise_for_status()
        return Dataset(client, **r.json())
    
    def load(self) -> pd.DataFrame:
        """Download and load dataset in-memory.
        
        Returns:
            pd.DataFrame: Pandas DataFrame.
        """
        endpoint = f"/datasets/{self.id}/download"
        r = self._client._api_request(endpoint, "get", stream=True)

        CHUNK_SIZE = 1024 * 1024 * 5

        buffer = io.BytesIO()
        for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
            buffer.write(chunk)

        buffer.seek(0)

        df = pd.read_parquet(buffer)
        self.columns = df.columns
        self.loc[:, :] = df

        return df

    def upload(self, df: pd.DataFrame, mode: _UploadModes = "append"):
        """Upload a Pandas data frame as dataset.
        
        Args:
            df: (pd.DataFrame): Data frame to upload.
            mode (str): Write mode, append to existing dataset or overwrite.

        Returns:
            bool: Result of uploading the dataset in chunks.
        """
        self._validate(df)

        MAX_MONO_UPLOAD = 1024 * 1024 * 5

        CHUNK_SIZE = MAX_MONO_UPLOAD

        data = self._to_binary(df)
        nr_bytes = len(data)

        total_chunks = math.ceil(nr_bytes / CHUNK_SIZE)

        # Initiate a new dataset upload
        upload = _DatasetUpload.find(self._client, self.id)
        if upload is None:
            upload = _DatasetUpload.init(self._client, self.id, mode)
        else:
            detail = "Upload for dataset %s is already pending" % self.name
            raise SulieError(_UPLOAD_IN_PROGRESS, detail)
            
        args = {
            "total": total_chunks,
            "desc": "Uploading dataset %s" % self.name,
            "unit_scale": True,
            "unit": "B"
        }

        with tqdm(**args) as progress:
            
            chunk = 1
            for i in range(0, nr_bytes, CHUNK_SIZE):
                upload_id = upload["upload_id"]
                endpoint = f"/datasets/{self.id}/uploads/{upload_id}"
                self._multipart_upload(
                    endpoint, 
                    data[i:i+CHUNK_SIZE], 
                    chunk, 
                    total_chunks
                )
                progress.update()
                chunk += 1
    
    def _multipart_upload(
            self,
            endpoint: str, 
            data: bytes, 
            chunk: int, 
            total: int
        ):
        """Upload a binary data in multiple chunks.
        
        Args:
            endpoint (str): API endpoint.
            data (bytes): Binary data.
            chunk (int): Chunk sequence that's being uploaded.
            total (int): Total number of chunks.
            size (int): Total size of the data in bytes.
        """
        files = {
            "bytes": ("bytes", data),
        }
        data = {
            "sequence": chunk,
            "total": total
        }
        r = self._client._api_request(endpoint, "post", files=files, data=data)
        return r.json()
    
    @staticmethod
    def _to_binary(df: pd.DataFrame) -> bytes:
        """Convert a Pandas DataFrame to binary Parquet format.
        
        Args:
            df (pd.DataFrame): Data frame to serialize.
        
        Returns:
            bytes: Binary content.
        """
        buffer = io.BytesIO()
        df.to_parquet(buffer, engine="pyarrow")
        buffer.flush()
        data = buffer.getvalue()
        return data


class _DatasetUpload:

    @staticmethod
    def find(
            client: Sulie, 
            dataset_id: str, 
            status: str = "pending"
        ) -> Dict[str, Any]:
        """Query database for an active dataset upload process.
        
        Args:
            client (Sulie): API client.
            dataset_id (str): ID of the dataset.
            status (str): Dataset status.

        Returns:
            dict: Dataset upload information.
        """
        params = {
            "status": status
        }
        endpoint = f"/datasets/{dataset_id}/uploads"
        r = client._api_request(endpoint, "get", params=params)
        data = r.json()
        return None if len(data) == 0 else data[0]
    
    @staticmethod
    def init(
            client: Sulie, 
            dataset_id: str, 
            mode: _UploadModes
        ) -> Dict[str, Any]:
        """Initiate a dataset upload.
        
        Args:
            client (Sulie): API client.
            mode (str): Upload mode.

        Returns:
            dict: Dataset upload information.
        """
        body = {
            "mode": mode
        }
        endpoint = f"/datasets/{dataset_id}/uploads"
        r = client._api_request(endpoint, "post", json=body)
        return r.json()
    

class Model:

    # Maximum context length
    max_context_len: int = 512

    # Maximum prediction length
    max_prediction_len: int = 64

    def __init__(
            self, 
            client: Sulie, 
            model_name: Optional[str] = None
        ) -> "Model":
        """Initialize a Model instance.
        
        Args:
            client (Sulie): API client.
            model_name (str): Name of the model.
        """
        self._client = client
        self.model_name = model_name or "mimosa-base"

    @staticmethod
    def list(client: Sulie) -> List["Model"]:
        """Query API for list of custom or fine-tuned models.
        
        Returns:
            list: List of models.
        """
        endpoint = "/deployments"
        r = client._api_request(endpoint, method="get")
        return [Model(client, d["model_name"]) for d in r.json()]

    @classmethod
    def get(cls, client: Sulie, model_name: str) -> Any:
        """Load a fine-tuned model variant.

        Args:
            client (Sulie): API client.
            model_name (str): Name of the model.

        Returns:
            Model: Model variant.
        """
        endpoint = "/deployments"
        params = {
            "model_name": model_name
        }
        r = client._api_request(endpoint, method="get", params=params)
        if r.status_code == 404:
            raise SulieError(_MODEL_NOT_FOUND, r.content.decode())
        
        r.raise_for_status()
        return Model(client, model_name)

    def forecast(
            self, 
            dataset: Union["Dataset", pd.DataFrame],
            target: str = None,
            group_by: str = None,
            date: str = None,
            aggr: str = None,
            frequency: Literal["H", "D", "W", "M", "Y"] = "D",
            horizon: int = 30,
            n_rows: int = None,
            **kwargs
        ) -> List[List[float]]:
        """Forecast using the time series foundation model.
        
        Args:
            dataset (Dataset): Dataset or pd.Dataframe.
            target (str): Name of the target value to forecast.
            group_by (str): Column to group the records on, optional.
            date (str): Name of the date (and time) column, optional.
            aggr (str): Column aggregation, required if `date` is set.
            frequency (str): Date aggregation frequency, optional.
            horizon (int): Prediction horizon length.
            n_rows (int): Number of rows to trim from tail.
            kwargs (dict): Optional inference keyword arguments.

        Returns:
            List: List of predictions.
        """
        assert isinstance(dataset, pd.DataFrame), "Unexpected data type"

        if isinstance(dataset, Dataset) and dataset.empty is True:
            dataset.load()

        if target not in dataset.columns:
            raise KeyError(f"Target column {target} not found in dataset")
        
        if group_by and group_by not in dataset.columns:
            raise KeyError(f"Group by column {group_by} not found in dataset")
        
        if date:
            if date not in dataset.columns:
                raise KeyError(f"Date column {date} not found in dataset")
        
            if aggr is None:
                raise ValueError("Aggregation function `aggr` required")
            
            dataset[date] = pd.to_datetime(dataset[date])
            dataset = dataset.sort_values(date)

            if frequency is not None:
                dataset = self._resample_dataset(
                    dataset, date, target, aggr, group_by, frequency
                )
        
        # Group data by group_by if specified
        if group_by is not None:
            grouped = dataset.groupby(group_by)

            forecasts = []
            for _, group in grouped:
                arr = group[target].values
                if n_rows is not None:
                    arr = arr[-n_rows:]

                r = self._call("forecast", arr, horizon, **kwargs)
                forecasts.append(r)
            return forecasts
        else:
            arr = dataset[target].values
            if n_rows is not None:
                arr = arr[-n_rows:]

            r = self._call("forecast", arr, horizon, **kwargs)
            return r

    def _resample_dataset(
            self, 
            dataset: Union["Dataset", pd.DataFrame], 
            date: str, 
            target: str, 
            aggr: str,
            group_by: str, 
            frequency: str
        ):
        """Resamples a DataFrame to a specified frequency.
        
        Args:
            dataset (Dataset): Dataset or pd.Dataframe with time series data.
            date (str): Name of the date and time column.
            target (str): Name of the target variable column.
            aggr (str): Aggregation function.
            group_by (str): Name of the grouping column.
            frequency (str): Desired resampling frequency.
        
        Returns:
            Dataset: Resampled Dataset.
        """
        offset = to_offset(frequency)

        agg_dict = {target: aggr}
        if group_by:
            agg_dict[group_by] = "first"
        
        # Resample and aggregate
        resampled = dataset.resample(offset, on=date).agg(agg_dict)
        
        return resampled.reset_index()
    
    def embed(self, arr: np.ndarray) -> List[Any]:
        """Extract encoder embeddings.
        
        Args:
            arr (np.ndarray): List of values, 1D or 2D array.

        Returns:
            List of embeddings.
        """
        r = self._call("embed", arr)
        return r

    def _call(
            self, 
            task: Literal["forecast", "embed"], 
            arr: np.ndarray,
            horizon: int = 30,
            **kwargs
        ) -> List[Any]:
        """Call the forecast API.
        
        Args:
            task (str): Inference task.
            arr (np.ndarray): Time series.
            horizon (int): Prediction horizon length.
            kwargs (dict): Optional inference parameters.

        Returns:
            list: List of forecasts.
        """
        if horizon > self.max_prediction_len:
            raise ValueError("Prediction length maximum is 64 data points")
        
        if len(arr.shape) == 1:
            arr = np.array([arr])

        if arr.shape[1] > self.max_context_len:
            msg = f"Context length {arr.shape[1]} > {self.max_context_len}"
            raise ValueError(msg)

        body = {
            "model_name": self.model_name,
            "context": arr.tolist(),
            "task": task,
            "prediction_length": horizon
        }
        
        for param in {"temperature", "top_k", "num_samples"}:
            if param in kwargs:
                body[param] = kwargs[param]

        r = self._client._api_request(f"/{task}", "post", json=body)
        if r.status_code == 426:
            raise SulieError(_UPGRADE_PLAN, "Too many requests, upgrade plan")
        
        r.raise_for_status()
        return r.json()


class FineTuneJob:

    def __init__(self, client: Sulie, **kwargs):
        self._client = client
        self.id = kwargs["job_id"]
        self.name = kwargs["job_name"]
        self.status = kwargs["status"]
        self.statistics = kwargs["statistics"]
        self.description = kwargs.get("description")

    @classmethod
    def fit(
            cls, 
            client: Sulie, 
            dataset_id: str, 
            target: str, 
            description: Optional[str] = None
        ) -> "FineTuneJob":
        """Fit a foundation time series model.
        
        Args:
            client (Sulie): API client.
            dataset_id (str): ID of the dataset the model will be fitted on.
            target (str): Name of the target value to forecast for.
            description (str): Description of the fine-tune job.

        Returns:
            FineTuneJob: Fine tune job.
        """
        # Start the fine-tune job
        data = {
            "dataset_id": dataset_id,
            "target": target,
            "description": description
        }
        r = client._api_request("/tune", method="post", json=data)
        r.raise_for_status()
        return FineTuneJob(client, **r.json())
    
    @classmethod
    def get(cls, client: Sulie, job_id: str) -> "FineTuneJob":
        """Describe a fine tune job.
        
        Args:
            client (Sulie): API client.
            job_id (str): ID of the fine tuning job.

        Returns:
            FineTuneJob: Fine tuning job.
        """
        endpoint = "/tune/%s" % job_id
        r = client._api_request(endpoint, method="get")
        r.raise_for_status()
        return FineTuneJob(client, **r.json())
    
    @staticmethod
    def list(client: Sulie) -> List["FineTuneJob"]:
        """List fine-tune jobs.
        
        Returns:
            list: List of fine-tune jobs.
        """
        endpoint = "/tune"
        r = client._api_request(endpoint, method="get")
        r.raise_for_status()
        return [FineTuneJob(client, **job) for job in r.json()]