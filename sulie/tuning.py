from .api import APIClient
from typing import List, Optional


class FineTuneJob:

    def __init__(self, client: APIClient, **kwargs):
        self._client = client
        self.id = kwargs["job_id"]
        self.name = kwargs["job_name"]
        self.status = kwargs["status"]
        self.statistics = kwargs["statistics"]
        self.description = kwargs.get("description")

    @classmethod
    def fit(
            cls, 
            client: APIClient, 
            dataset_id: str, 
            target: str, 
            group_by: Optional[str] = None,
            description: Optional[str] = None
        ) -> "FineTuneJob":
        """Fit a foundation time series model.
        
        Args:
            client (APIClient): API client.
            dataset_id (str): ID of the dataset the model will be fitted on.
            target_col (str): Name of the target value to forecast for.
            id_col (str): Name of the series to group the dataset by.
            description (str): Description of the fine-tune job.

        Returns:
            FineTuneJob: Fine tune job.
        """
        data = {
            "dataset_id": dataset_id,
            "target": target,
            "description": description,
            "group_by": group_by
        }
        r = client.request("/tune", method="post", json=data)
        r.raise_for_status()
        return FineTuneJob(client, **r.json())
    
    @classmethod
    def get(cls, client: APIClient, job_id: str) -> "FineTuneJob":
        """Describe a fine tune job.
        
        Args:
            client (APIClient): API client.
            job_id (str): ID of the fine tuning job.

        Returns:
            FineTuneJob: Fine tuning job.
        """
        endpoint = "/tune/%s" % job_id
        r = client.request(endpoint, method="get")
        r.raise_for_status()
        return FineTuneJob(client, **r.json())
    
    @staticmethod
    def list(client: APIClient) -> List["FineTuneJob"]:
        """List fine-tune jobs.
        
        Returns:
            list: List of fine-tune jobs.
        """
        endpoint = "/tune"
        r = client.request(endpoint, method="get")
        r.raise_for_status()

        jobs = []
        for data in r.json():
            job = FineTuneJob(client, **data)
            jobs.append(job)
        
        return jobs