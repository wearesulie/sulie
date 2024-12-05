import io
import math
import pandas as pd
from .api import APIClient
from .errors import DATASET_NOT_FOUND, UPLOAD_IN_PROGRESS, SulieError
from typing import Any, Dict, Literal
from tqdm import tqdm


# Supported dataset upload types
UploadModes = Literal["append", "overwrite"]


class Dataset(pd.DataFrame):
    
    def __init__(self, client: APIClient, **kwargs):
        super().__init__()
        self._client = client
        self.name = kwargs["name"]
        self.description = kwargs["description"]
        self.id = kwargs["dataset_id"]

    def __repr__(self):
        return f"{self.id} {self.name} {self.description}"

    @classmethod
    def _from_dict(cls, client: APIClient, **kwargs):
        """Initialize a Dataset instance from keyword arguments.
        
        Args:
            client (APIClient): API client.
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
    def get(client: APIClient, name: str) -> Dict[str, Any]:
        """Query API for dataset metadata.
        
        Args:
            name (str): Name of the dataset.

        Returns:
            dict: Dataset metadata.
        """
        params = {
            "name": name
        }
        r = client.request("/datasets", method="get", params=params)
        if r.status_code == 404:
            detail = "Dataset %s does not exist" % name
            raise SulieError(DATASET_NOT_FOUND, detail)
        
        return Dataset(client, **r.json()[0])

    @staticmethod
    def create(
            client: APIClient, 
            name: str, 
            description: str, 
            **kwargs
        ) -> Dict[str, Any]:
        """Create a dataset.
        
        Args:
            client (APIClient): API client.
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
        r = client.request("/datasets", "post", json=body)
        r.raise_for_status()
        return Dataset(client, **r.json())
    
    def load(self) -> pd.DataFrame:
        """Download and load dataset in-memory.
        
        Returns:
            pd.DataFrame: Pandas DataFrame.
        """
        endpoint = f"/datasets/{self.id}/download"
        r = self._client.request(endpoint, "get", stream=True)

        CHUNK_SIZE = 1024 * 1024 * 5

        buffer = io.BytesIO()
        for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
            buffer.write(chunk)

        buffer.seek(0)

        df = pd.read_parquet(buffer)
        for column in df.columns:
            self[column] = df[column]
        
        self.columns = df.columns
    
        return df

    def upload(self, df: pd.DataFrame, mode: UploadModes = "append"):
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
        upload = DatasetUpload.find(self._client, self.id)
        if upload is None:
            upload = DatasetUpload.init(self._client, self.id, mode)
        else:
            detail = "Upload for dataset %s is already pending" % self.name
            raise SulieError(UPLOAD_IN_PROGRESS, detail)
            
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
        r = self._client.request(endpoint, "post", files=files, data=data)
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


class DatasetUpload:

    @staticmethod
    def find(
            client: APIClient, 
            dataset_id: str, 
            status: str = "pending"
        ) -> Dict[str, Any]:
        """Query database for an active dataset upload process.
        
        Args:
            client (APIClient): API client.
            dataset_id (str): ID of the dataset.
            status (str): Dataset status.

        Returns:
            dict: Dataset upload information.
        """
        params = {
            "status": status
        }
        endpoint = f"/datasets/{dataset_id}/uploads"
        r = client.request(endpoint, "get", params=params)
        data = r.json()
        return None if len(data) == 0 else data[0]
    
    @staticmethod
    def init(
            client: APIClient, 
            dataset_id: str, 
            mode: UploadModes
        ) -> Dict[str, Any]:
        """Initiate a dataset upload.
        
        Args:
            client (APIClient): API client.
            mode (str): Upload mode.

        Returns:
            dict: Dataset upload information.
        """
        body = {
            "mode": mode
        }
        endpoint = f"/datasets/{dataset_id}/uploads"
        r = client.request(endpoint, "post", json=body)
        return r.json()
