import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

from .api import APIClient
from .datasets import Dataset
from .errors import MODEL_NOT_FOUND, INTERNAL_ERROR, UPGRADE_PLAN, SulieError
from .metrics import mae, wape, weighted_quantile_loss
from .utils import array_split
from dataclasses import dataclass
from pandas.tseries.frequencies import to_offset
from typing import Any, List, Literal, Optional, Union
from tqdm import tqdm


@dataclass
class Forecast:
    """A class for time series forecasting with confidence intervals."""
    context: List[Union[int, float]]
    median: List[float]
    quantiles: List[List[float]]

    def __post_init__(self):
        """Validate input data and store bounds."""
        self._validate_inputs()
        self.lower_bound = self.quantiles[0]
        self.upper_bound = self.quantiles[-1]

    def _validate_inputs(self) -> None:
        """Validate input data dimensions."""
        if len(self.quantiles) < 2:
            raise ValueError("At least two quantile sequences are required")
        
        forecast_length = len(self.median)
        if not all(len(q) == forecast_length for q in self.quantiles):
            raise ValueError("Quantiles must have same length as median")

    def plot(self, height: int = 4, width: int = 8) -> None:
        """Display actual data, predicted forecasts, and confidence interval.
        
        Args:
            height: Plot height in inches. Defaults to 4.
            width: Plot width in inches. Defaults to 8.
        """
        context_size = len(self.context)
        horizon_length = len(self.median)
        forecast_indices = range(context_size, context_size + horizon_length)

        plt.figure(figsize=(width, height))
        
        # Plot historical data
        plt.plot(
            self.context, 
            color="royalblue", 
            label="Historical data"
        )
        
        # Plot median forecast
        plt.plot(
            forecast_indices, 
            self.median, 
            color="green", 
            label="Median forecast"
        )
        
        # Plot confidence interval
        plt.fill_between(
            forecast_indices,
            self.lower_bound,
            self.upper_bound,
            color="tomato",
            alpha=0.3,
            label="Prediction interval"
        )
        
        plt.legend()
        plt.grid(True)
        plt.show()


class Model:

    # Maximum context length
    MAX_CONTEXT_LEN: int = 512

    # Maximum prediction length
    MAX_PREDICTION_LEN: int = 64

    # Maximum number of groups to forecast on in a single pass
    MAX_NGROUPS: int = 100

    def __init__(
            self, 
            client: APIClient, 
            model_name: Optional[str] = None
        ) -> "Model":
        self._client = client
        self.model_name = model_name or "mimosa-base"

    @staticmethod
    def list(client: APIClient) -> List["Model"]:
        """Query API for list of custom or fine-tuned models.

        Args:
            client (APIClient): API client.
        
        Returns:
            list: List of models.
        """
        endpoint = "/deployments"
        r = client.request(endpoint, method="get")

        models = []
        for data in r.json():
            model = Model(client, data["model_name"])
            models.append(model)
        
        return models

    @classmethod
    def get(cls, client: APIClient, model_name: str) -> Any:
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
        r = client.request(endpoint, method="get", params=params)
        if r.status_code == 404:
            raise SulieError(MODEL_NOT_FOUND, r.content.decode())
        
        r.raise_for_status()
        return Model(client, model_name)
    
    def evaluate(
            self, 
            dataset: Union[Dataset, pd.DataFrame],
            target_col: str,
            horizon: int = 30,
            id_col: str = None,
            metric: Literal["MAE", "WQL", "WAPE"] = "MAE",
            metric_aggregation: Literal["mean", "median"] = "mean",
            iterations: int = 100,
        ) -> float:
        """Evaluate the model's performance on a user-provided dataset.

        This method assesses the model using a specified evaluation metric 
        by applying sliding windows to randomly shuffle the data. The result 
        is calculated based on either Weighted Quantile Loss (WQL) or 
        Weighted Absolute Percentage Error (WAPE).

        Args:
            dataset (Union[Dataset, pd.DataFrame]): Input dataset to forecast on.
            target_col (str): Target column to forecast.
            horizon (int): Number of future steps to predict.
            id_col (str, optional): Column for multiple time series IDs.
            metric (Literal["MAE", "WQL", "WAPE"]): Name of the metric.
            metric_aggregation (Literal["mean", "median"]): Function name.
            iterations (int): Number of evaluation iterations.
        """
        MIN_SAMPLES = 64

        if target_col not in dataset:
            raise ValueError(f"A minimum of {MIN_SAMPLES} is required")
        
        if iterations < 0:
            raise ValueError(f"Number of iterations must be > 0")
        
        if not hasattr(np, metric_aggregation):
            raise NotImplementedError(f"Unknown function {metric_aggregation}")
        
        # Validate target shape
        if id_col is not None:
            groups = []
            for _, group in dataset.groupby(id_col):
                if group.shape[0] >= MIN_SAMPLES:
                    groups.append(group)

            # Reconstruct DataFrame with only valid groups
            dataset = pd.concat(groups)
        else:
            if dataset.shape[0] < MIN_SAMPLES:
                detail = f"Dataset must have at least {MIN_SAMPLES} samples"
                raise ValueError(detail)

        args = {
            "total": iterations,
            "desc": f"Evaluating model {self.model_name}",
            "unit_scale": True,
            "unit": "B"
        }
        with tqdm(**args) as progress:
            
            scores = []
            for _ in range(iterations):
                # Get the target array based on id_col if provided
                if id_col is not None:
                    selected_group = random.choice(dataset[id_col].unique())
                    target_arr = dataset[
                        dataset[id_col] == selected_group
                    ][target_col]
                else:
                    target_arr = dataset[target_col]
                
                # Calculate available length considering horizon
                nr_samples = target_arr.shape[0] - horizon
                
                # Skip this iteration if we don't have enough samples
                if nr_samples < MIN_SAMPLES:
                    progress.update()
                    continue 
                
                # Determine context length
                max_possible_context = min(nr_samples, self.MAX_CONTEXT_LEN)
                rand_context_length = random.randint(
                    MIN_SAMPLES, max_possible_context
                )
                context_length = min(max_possible_context, rand_context_length)
                
                # Calculate valid start index range
                max_start_idx = nr_samples - context_length
                
                # Skip if we can't find a valid window
                if max_start_idx < 0:
                    progress.update()
                    continue 
                
                # Randomly select start index and generate indices
                start_idx = random.randint(0, max_start_idx)
                indices = list(range(start_idx, start_idx + context_length))

                # Randomly sample by prediction_length
                sampled = np.nan_to_num(target_arr.values[indices], nan=0)
                if isinstance(sampled, np.ndarray) and len(sampled.shape) == 1:
                    sampled = [sampled]
                    
                forecast = self._call("forecast", sampled, horizon)

                start, end = indices[-1], indices[-1] + horizon
                actual = target_arr[start:end]

                functions = {
                    "MAE": mae,
                    "WQL": weighted_quantile_loss,
                    "WAPE": wape
                }
                if metric not in functions:
                    raise NotImplementedError(f"Metric {metric} not supported")
                
                score = functions[metric](actual, forecast.median) 
                scores.append(score)

                progress.update()
        
        return getattr(np, metric_aggregation)(scores)

    def forecast(
            self, 
            dataset: Union[Dataset, pd.DataFrame],
            target_col: str = "y",
            horizon: int = 7,
            id_col: str = None,
            timestamp_col: str = None,
            aggr: str = None,
            frequency: Literal["H", "D", "W", "M", "Y"] = None,
            quantiles: List[float] = [0.1, 0.9]
        ) -> Union[Forecast, List[Forecast]]:
        """Generate probabilistic forecasts using the foundation model.

        For a single time series, returns one Forecast object containing the
        historical context, median forecast, and prediction intervals. For 
        multiple series (when id_col is specified), returns a list of Forecast 
        objects.

        Args:
            dataset (Union[Dataset, pd.DataFrame]): Input time series data.
            target_col (str): Target column to forecast.
            horizon (int): Number of future steps to predict.
            id_col (str, optional): Column for multiple time series IDs.
            timestamp_col (str): Column containing timestamps.
            aggr (str, optional): Temporal aggregation function.
            frequency (Literal["H","D","W","M","Y"]): Time step frequency.
            quantiles (List[float]): Probability levels for intervals.

        Returns:
            Union[Forecast, List[Forecast]]: Single Forecast object if id_col is
                None, otherwise list of Forecast objects (one per unique ID)

        Raises:
            ValueError: If timestamp_col specified but aggr is None
            ValueError: If specified columns not found in dataset
            ValueError: If frequency is invalid
        """
        assert isinstance(dataset, pd.DataFrame), "Unexpected data type"

        if isinstance(dataset, Dataset) and dataset.empty is True:
            dataset.load()

        if target_col not in dataset.columns:
            raise KeyError(f"Target column {target_col} not found in dataset")
        
        if id_col and id_col not in dataset.columns:
            raise KeyError(f"ID column {id_col} not found in dataset")
        
        if timestamp_col:
            if timestamp_col not in dataset.columns:
                detail = f"Timestamp column {timestamp_col} not in dataset"
                raise KeyError(detail)
            
            dataset[timestamp_col] = pd.to_datetime(dataset[timestamp_col])
            dataset = dataset.sort_values(timestamp_col)

            if frequency is not None:
                if aggr is None:
                    raise ValueError("Aggregation function `aggr` required")
            
                dataset = self._resample_dataset(
                    dataset=dataset,
                    timestamp_col=timestamp_col,
                    target_col=target_col,
                    aggr=aggr,
                    id_col=id_col,
                    frequency=frequency
                )
        
        if id_col is None:
            series = [dataset[target_col].values]
        else:
            groups = dataset.groupby(id_col)

            series = []
            for _, group in groups:
                if timestamp_col is not None:
                    group = group.sort_values(timestamp_col)
                
                series.append(group[target_col].values)

        if len(series) < Model.MAX_NGROUPS:
            return self._call("forecast", series, horizon, quantiles)
        
        # Optionaly forecast in batches
        r = []
        for batch in array_split(series, n=100):
            r += self._call("forecast", batch, horizon, quantiles)
        
        return r

    def _resample_dataset(
            self, 
            dataset: Union["Dataset", pd.DataFrame], 
            timestamp_col: str, 
            target_col: str, 
            aggr: str,
            id_col: str, 
            frequency: str
        ):
        """Resamples a DataFrame to a specified frequency.
        
        Args:
            dataset (Dataset): Dataset or pd.Dataframe with time series data.
            timestamp_col (str): Name of the date and time column.
            target_col (str): Name of the target variable column.
            aggr (str): Aggregation function.
            id_col (str): Name of the grouping column.
            frequency (str): Desired resampling frequency.
        
        Returns:
            Dataset: Resampled Dataset.
        """
        offset = to_offset(frequency)

        agg_dict = {target_col: aggr}
        if id_col:
            agg_dict[id_col] = "first"
        
        # Resample and aggregate
        resampled = dataset.resample(offset, on=timestamp_col).agg(agg_dict)
        
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
            series: List[np.ndarray],
            horizon: int = 30,
            quantiles: List[float] = [0.1, 0.9]
        ) -> List[Any]:
        """Call the forecast API.
        
        Args:
            task (str): Inference task.
            series (list): Time series.
            horizon (int): Prediction horizon length.
            quantiles (List[float]): Probability levels for intervals.

        Returns:
            list: List of forecasts.
        """
        if horizon > Model.MAX_PREDICTION_LEN:
            raise ValueError("Prediction length maximum is 64 data points")
        
        if len(series) == 0:
            raise ValueError("At least 1 series is required")
        
        # Determine smallest series
        min_n_rows = series[0].shape[0]
        for idx, Y in enumerate(series):
            if Y.shape[0] < min_n_rows:
                min_n_rows = Y.shape[0]

        if min_n_rows > Model.MAX_CONTEXT_LEN:
            min_n_rows = Model.MAX_CONTEXT_LEN

        for idx, Y in enumerate(series):
            Y = Y[-min_n_rows:]
            series[idx] = Y.tolist()

        body = {
            "model_name": self.model_name,
            "context": series,
            "task": task,
            "prediction_length": horizon,
            "sdk_version": "1.0.8"
        }

        r = self._client.request(f"/{task}", "post", json=body)
        if r.status_code == 426:
            raise SulieError(UPGRADE_PLAN, "Too many requests, upgrade plan")
        elif r.status_code >= 400:
            raise SulieError(INTERNAL_ERROR, r.content.decode())

        body = r.json()    
        if task == "embed":
            return body

        forecasts = []
        for idx, (median, quantiles) in enumerate(body):
            forecast = Forecast(series[idx], median, quantiles)
            forecasts.append(forecast)

        return forecasts if len(forecasts) > 1 else forecasts[0]