
<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://y1bix39g-cdn-default.s3.us-east-1.amazonaws.com/assets/sulie-icon-white.svg">
    <img alt="Sulie logo" src="https://y1bix39g-cdn-default.s3.us-east-1.amazonaws.com/assets/sulie-bw-sign.svg" width="50%">
  </picture>
</div>


<p align="center">
  <a href="https://docs.sulie.co">
    <img src="https://img.shields.io/badge/docs-mintlify-blue" alt="docs_badge">
  </a>
  <a href="https://pypi.org/project/sulie/">
    <img src="https://img.shields.io/pypi/v/sulie.svg" alt="PyPI Badge">
  </a>
</p>

### Sulie - Foundation Models for Time-Series Forecasting

Sulie offers cutting-edge foundation models for time series forecasting, enabling accurate, **zero-shot predictions** with minimal setup. Our transformer-based models automate the process, eliminating the need for manual training and complex configurations. 

<p align="center">
    <a href="https://docs.sulie.co">Documentation</a>
    ·
    <a href="https://github.com/wearesulie/sulie/issues/new">Report Bug</a>
    ·
  <a href="https://join.slack.com/t/sulie-community/shared_invite/zt-2tpeh8opw-vFbpmTrckMWlcQ2OvLCTXA">Join Our Slack</a>
    ·
    <a href="https://twitter.com/wearesulie">Twitter</a>
  </p>

## 🔥 Features

* __Zero-Shot Forecasting__: Obtain precise forecasts instantly with our foundation model, without requiring training or preprocessing of historical data.
* __Auto Fine-Tuning__: Enhance model performance with a single API call. We manage the entire training pipeline, providing transparency into model selection and metrics.
* __Covariates Support__ (Enterprise): Conduct multivariate forecasting by incorporating dynamic and static covariates with no feature engineering needed.
* __Managed Infrastructure__: Focus on forecasting as we manage all aspects of deployment, scaling, and maintenance seamlessly.
* __Centralized Datasets__: Push time series data continuously through our Python SDK, creating a centralized, versioned repository accessible across your organization.

## 🚀 Getting Started

To begin using the Sulie SDK, you’ll need an API key, which can be generated from the **Sulie Dashboard**:

1. Visit the [Sulie Dashboard](https://app.sulie.co).
2. Sign in to your Sulie account.
3. Navigate to the **API Keys** section.
4. Generate a new API key and copy it to use within the SDK.

With your API key ready, you’re set to start forecasting.

## ⚙️ Installation

To install the Sulie SDK, simply run:

```bash
pip install sulie==1.0.6
```

## Quick Start Example

After installation, initialize the SDK using your API key to start forecasting with Mimosa:

```python
from sulie import Sulie

# Initialize the Sulie client
client = Sulie(api_key="YOUR_API_KEY")
```

## ⚡️ Features

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

The `Forecast` object includes three lists: `low`, `median`, and `high`, corresponding to different certainty levels in the predictions. These help you understand the range of possible outcomes, from conservative to optimistic.

You can also visualize the forecasts directly by calling the plot function:
```python
forecast.plot()
```

This quickly generates a chart showing the forecast ranges, making it easy to spot trends and variability in the results. Perfect for a fast, clear analysis.

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
| `group_by`    | Name of the column to group the DataFrame series by.| `None`  |
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

> [!NOTE] 
> Datasets are an optional feature. To make forecasts or even fine-tune a foundation model, you may also pass a Pandas `DataFrame` to the `forecast` and `fine_tune` functions.

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

### 📚 Additional Resources
- **[API Documentation](https://docs.sulie.co)**: Full documentation with detailed usage.
- **[Forecasting Guide](https://docs.sulie.co/capabilities/forecasting)**: Detailed parameters for Mimosa forecasting.
- **[Fine-Tuning Guide](https://docs.sulie.co/capabilities/fine-tuning)**: Options and tuning customization.
- **[Support](mailto:support@sulie.co)**: Assistance and feedback on the SDK.
