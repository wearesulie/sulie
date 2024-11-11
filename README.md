### Sulie - Managed Time Series Forecasting with Mimosa Foundation Model

The **Sulie SDK** offers seamless integration with the Sulie platform for advanced time series forecasting powered by **Mimosa**—a transformer-based foundation model optimized specifically for time series data. Mimosa provides high accuracy for **zero-shot forecasting** and **automatic fine-tuning**, enabling tailored forecasts without extensive pre-training. From deployment to scaling, we handle the MLOps heavy lifting, so you can focus on making use of the forecasts.

## Getting Started

To begin using the Sulie SDK, you’ll need an API key, which can be generated from the **Sulie Dashboard**:

1. Visit the [Sulie Dashboard](https://app.sulie.co).
2. Sign in to your Sulie account.
3. Navigate to the **API Keys** section.
4. Generate a new API key and copy it to use within the SDK.

With your API key ready, you’re set to start forecasting.

## Installation

To install the Sulie SDK, simply run:

```bash
pip install sulie
```

## Quick Start Example

After installation, initialize the SDK using your API key to start forecasting with Mimosa:

```python
from sulie import Sulie

# Initialize the Sulie client
client = Sulie(api_key="YOUR_API_KEY")
```

## Features

### 1. Forecasting with Mimosa
Generate accurate time series forecasts using Mimosa’s **zero-shot inference** capabilities. This approach is ideal when you need fast, reliable predictions without training the model.

```python
import pandas as pd

# Example time series data
df = pd.DataFrame({
    'timestamp': pd.date_range(start='2023-01-01', periods=1000, freq='H'),
    'demand': [ ... ],           # Demand data
    'location': ['Plant A', ...] # Data for different locations
})

# Forecast demand for each location over the next 24 hours
forecast = client.forecast(
    dataset=df,
    target='demand',
    group_by='location',
    date='timestamp',
    frequency='H',
    horizon=24,            # Predict 24 hours ahead
    num_samples=100        # Generate probabilistic forecasts
)
print(forecast)
```

#### Forecasting Parameters

| Name          | Description                                              | Default     |
|---------------|----------------------------------------------------------|-------------|
| `dataset`     | A `Dataset` or `pd.DataFrame` containing time series data.| **Required**|
| `target`      | Column name for the forecast variable.                    | **Required**|
| `group_by`    | Column name to group data by (e.g., different locations). | `None`      |
| `date`        | Timestamp column name.                                    | `None`      |
| `frequency`   | Frequency of the time series (e.g., `H` for hourly).      | `None`      |
| `horizon`     | Time steps to forecast ahead.                             | `24`        |
| `num_samples` | Number of probabilistic forecast samples.                 | `100`       |

### 2. Fine-Tuning for Customized Forecasting
With automatic fine-tuning, you can optimize Mimosa for unique datasets and business cases. The fine-tuning process uses **Weighted Quantile Loss (WQL)** for evaluation, ensuring high accuracy.

```python
# Fine-tune Mimosa on custom dataset
fine_tune_job = client.fine_tune(
    dataset=df,
    target="demand",
    description="Fine-tune for Plant A demand prediction"
)

# Check the fine-tuning job status
print(f"Job status: {fine_tune_job.status}")
```

#### Fine-Tuning Parameters

| Name          | Description                                         | Default |
|---------------|-----------------------------------------------------|---------|
| `dataset`     | A `Dataset` or `pd.DataFrame` with time series data.| Required|
| `target`      | Target variable for optimization.                   | Required|
| `description` | Description of the fine-tuning job.                 | `None`  |

Once fine-tuning completes, the model is automatically deployed and available for forecasting.

### 3. Managing Datasets
Sulie’s Dataset API lets you manage and version your datasets, making them accessible for forecasting and fine-tuning across teams.

```python
# Upload a dataset to Sulie
dataset = client.upload_dataset(
    name="product-sales-data-v1",
    df=df,
    mode="append"  # Choose 'append' or 'overwrite'
)

# List available datasets
datasets = client.list_datasets()
print(f"Available datasets: {datasets}")
```

#### Dataset Management Functions
- **Upload**: Store and version your data for easy access and updates.
- **List**: Retrieve a list of uploaded datasets.
- **Update**: Append or overwrite data for an existing dataset.

### 4. Forecasting with Custom Models
Fine-tuned models can be selected for new forecasts using `list_custom_models` or `get_model`.

```python
# List custom and fine-tuned models
custom_models = client.list_custom_models()

# Select and forecast with a fine-tuned model
model_name = custom_models[0].name
custom_model = client.get_model(model_name)

# Forecast using the selected model
forecast_custom = custom_model.forecast(
    dataset=df,
    target='demand',
    group_by='location',
    date='timestamp',
    frequency='H',
    horizon=24,
    num_samples=50
)
print(forecast_custom)
```

---

### Additional Resources
- **[API Documentation](https://docs.sulie.co)**: Full documentation with detailed usage.
- **[Forecasting Guide](https://docs.sulie.co/capabilities/forecasting)**: Detailed parameters for Mimosa forecasting.
- **[Fine-Tuning Guide](https://docs.sulie.co/capabilities/fine-tuning)**: Options and tuning customization.
- **[Support](mailto:support@sulie.co)**: Assistance and feedback on the SDK.
