import requests
from typing import Literal


class APIClient:

    def __init__(self, api_url: str, api_key: str):
        """Initialize an API client."""
        self.api_url = api_url
        self.api_key = api_key
        self._caller = self._validate_api_key()["organization_id"]

    def _validate_api_key(self):
        """Validate API key.
        
        Raises:
            ValueError: If API key is invalid or expired.
        """
        r = self.request("/keys", "put", headers={"Api-Key": self.api_key})
        r.raise_for_status()
        return r.json()

    def request(
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
            message = f"Authentication failed, invalid API key for {action}."
        else:
            message = f"Unexpected error during {action}."

        data = response.json()
        error_msg = data.get("error", "No error details provided")

        message += error_msg

        raise requests.exceptions.HTTPError(message, response=response)