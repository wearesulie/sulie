import logging
import numpy as np
import os
import pandas as pd
import uuid

from .api import APIClient
from .datasets import Dataset, UploadModes
from .errors import DATASET_NOT_FOUND, SulieError
from .inference import Model, Forecast
from .tuning import FineTuneJob
from typing import Optional, List, Any, Literal, Union

__version__ = "1.0.8"

logger = logging.getLogger("sulie")

_DEFAULT_API_URL = "https://api.sulie.co"


class Sulie:
    """Client for interacting with Sulie's time series forecasting API.

    This class provides methods to manage datasets, generate forecasts,
    fine-tune models, and evaluate model performance on time series data.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Sulie client with API credentials.

        Args:
            api_key (str, optional): API key for authentication. If None, 
                reads from SULIE_API_KEY environment variable
        
        Raises:
            ValueError: If no API key is provided or found in environment
        """
        api_url = os.environ.get("SULIE_API_URL") or _DEFAULT_API_URL
        
        api_key = api_key or os.environ.get("SULIE_API_KEY")
        if api_key is None:
            raise ValueError("Unspecified `api_key` configuration")
        
        self._client = APIClient(api_url, api_key)

    def list_datasets(self) -> List[Dataset]:
        """List available datasets for the organization the API key is 
        associated with.

        Returns:
            List[Dataset]: Collection of available datasets.
        """
        r = self._client.request("/datasets", "get")
        r.raise_for_status()

        datasets = []
        for kwargs in r.json():
            dataset = Dataset(self._client, **kwargs)
            datasets.append(dataset)

        return datasets

    def get_dataset(self, name: str) -> Dataset:
        """Retrieve a specific dataset by name.

        Args:
            name (str): Name of the dataset to retrieve.

        Returns:
            Dataset: Requested dataset.

        Raises:
            SulieError: If dataset not found.
        """
        dataset = Dataset.get(self._client, name)
        return dataset

    def upload_dataset(
            self, 
            df: pd.DataFrame,
            name: str, 
            mode: UploadModes = "append", 
            **kwargs
        ) -> Dataset:
        """Upload a dataframe as a new dataset or append to existing one.

        Args:
            df (pd.DataFrame): Data to upload
            name (str): Name for the dataset
            mode (UploadModes): How to handle existing data, default "append".
            **kwargs: Additional arguments like description

        Returns:
            Dataset: Created or updated dataset
        """
        try:
            dataset: Dataset = Dataset.get(self._client, name)
        except SulieError as e:
            if e.code != DATASET_NOT_FOUND:
                raise
            
            desc = kwargs.get("description")
            storage = kwargs.get("storage", "persisted")

            dataset = Dataset.create(self._client, name, desc, mode=storage)
        finally:
            dataset.upload(df, mode)
            return dataset

    def list_custom_models(self) -> List[Model]:
        """List custom fine-tuned models.

        Returns:
            List[Model]: Collection of available custom models.
        """
        return Model.list(self._client)

    def get_model(self, model_name: str) -> Model:
        """Retrieve a specific custom model by name.

        Args:
            model_name (str): Name of the model.

        Returns:
            Model: Requested model.

        Raises:
            SulieError: If model not found.
        """
        return Model.get(self._client, model_name)

    def evaluate(
            self, 
            dataset: Union[Dataset, pd.DataFrame],
            target_col: str,
            horizon: int = 30,
            id_col: str = None,
            metric: Literal["WQL", "WAPE"] = "WQL",
            metric_aggregation: Literal["mean", "median"] = "mean",
            iterations: int = 100,
            model: Optional[Model] = None
        ) -> float:
        """Evaluate model performance on time series data.

        Args:
            dataset (Union[Dataset, pd.DataFrame]): Input dataset to forecast on.
            target_col (str): Target column to forecast.
            horizon (int): Number of future steps to predict.
            id_col (str, optional): Column for multiple time series IDs.
            metric (Literal["WQL", "WAPE"]): Evaluation metric
            metric_aggregation (Literal["mean", "median"]): How to aggregate
            iterations (int): Number of evaluation runs
            model (Model, optional): Model to evaluate, uses default if None

        Returns:
            float: Aggregated evaluation score
        """
        model: Model = model or Model(self._client)
        score = model.evaluate(
            dataset=dataset,
            target_col=target_col,
            horizon=horizon,
            id_col=id_col,
            metric=metric,
            metric_aggregation=metric_aggregation,
            iterations=iterations
        )
        return score

    def forecast(
            self, 
            dataset: Union[Dataset, pd.DataFrame],
            target_col: str = "y",
            horizon: int = 7,
            id_col: Optional[str] = None,
            timestamp_col: str = None,
            aggr: Optional[str] = None,
            frequency: Literal["H", "D", "W", "M", "Y"] = None,
            model: Optional[Model] = None,
            quantiles: List[float] = [0.1, 0.9]
        ) -> Union[Forecast, List[Forecast]]:
        """Generate probabilistic forecasts using time series model.

        Args:
            dataset (Union[Dataset, pd.DataFrame]): Input time series data.
            target_col (str): Column containing values to forecast.
            horizon (int): Number of future steps to predict.
            id_col (str, optional): Column for multiple time series IDs.
            timestamp_col (str): Column containing timestamps.
            aggr (str, optional): Temporal aggregation function.
            frequency (Literal["H","D","W","M","Y"]): Time step frequency.
            model (Model, optional): Model to use, defaults to base model.
            quantiles (List[float]): Probability levels for intervals.

        Returns:
            Union[Forecast, List[Forecast]]: Forecasts for each series.

        Raises:
            ValueError: If timestamp_col specified but aggr is None.
            ValueError: If specified columns not found in dataset.
            ValueError: If frequency is invalid.
        """
        model: Model = model or Model(self._client)
        r = model.forecast(
            dataset=dataset, 
            target_col=target_col, 
            id_col=id_col, 
            timestamp_col=timestamp_col, 
            aggr=aggr, 
            frequency=frequency, 
            horizon=horizon,
            quantiles=quantiles
        )
        return r

    def embed(
            self,
            arr: np.ndarray,
            model: Optional[Model] = None
        ) -> List[Any]:
        """Extract embeddings from time series data.

        Args:
            arr (np.ndarray): Input time series data (1D or 2D array).
            model (Model, optional): Model to use, defaults to base model.

        Returns:
            List[Any]: Extracted embeddings.
        """
        model: Model = model or Model(self._client)
        r = model.embed(arr)
        return r

    def fine_tune(
            self,
            dataset: Union[Dataset, pd.DataFrame],
            target_col: str,
            id_col: Optional[str] = None,
            description: Optional[str] = None
        ) -> FineTuneJob:
        """Create a new fine-tuning job for the model based on a custom dataset.

        Args:
            dataset (Union[Dataset, pd.DataFrame]): Training data.
            target_col (str): Column containing target values.
            id_col(str, optional): Column for grouping multiple series.
            description (str, optional): Description of fine-tuning job.

        Returns:
            FineTuneJob: Created fine-tuning job.

        Raises:
            ValueError: If dataset has fewer than 1000 samples.
            KeyError: If target column not found in dataset.
        """
        if isinstance(dataset, Dataset) and dataset.empty is True:
            dataset.load()

        if target_col not in dataset.columns:
            raise KeyError(f"Target column {target_col} not found in dataset")

        min_dataset_size = 1000
        if len(dataset[target_col]) < min_dataset_size:
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
        job = FineTuneJob.fit(
            client=self._client, 
            dataset_id=train_dataset.id, 
            target=target_col, 
            group_by=id_col, 
            description=description
        )
        return job

    def list_fine_tuning_jobs(self) -> List[FineTuneJob]:
        """List all fine-tuning jobs for the organization.

        Returns:
            List[FineTuneJob]: Collection of fine-tuning jobs.
        """
        return FineTuneJob.list(self._client)