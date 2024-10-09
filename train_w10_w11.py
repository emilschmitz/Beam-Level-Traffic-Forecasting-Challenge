# %%
# Linear Regression and XGBoost Stacking for Traffic Forecasting

# %% [markdown]
# # Linear Regression and XGBoost Stacking
# In this script, we will:
# - Fit a **Linear Regression** model to each beam's throughput volume (`thp_vol`) using only **time** as the feature.
# - Calculate the **residuals**, scaled down by the linear predictions.
# - Fit an **XGBoost** model to these residuals using engineered features.
# - Combine the predictions from both models to get the final prediction.
# 
# This method is an example of **Model Stacking** or **Residual Modeling**, where we use one model to capture the main trend and another to model the residuals.

# %% [markdown]
# ## Imports and Setup

# %%
import os
import logging
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import wandb
import yaml
import matplotlib.pyplot as plt

# Load utility functions (assuming utils.py is in the same directory)
import utils

# Manually set the notebook name for W&B
os.environ["WANDB_NOTEBOOK_NAME"] = "train_w10_w11.ipynb"

# Set up logging
logging.basicConfig(level=logging.INFO)

# %% [markdown]
# ## Configuration

# %%
# Set DEBUG mode
DEBUG = False

# Load configuration from YAML file
config_file_path = Path('configs') / 'seasonality_config.yaml'
with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)

# Extract XGBoost hyperparameters from config
xgb_hyperparams = config.get('xgb_hyperparams', {})

# %% [markdown]
# ## Data Loading

# %%
# Read the CSV files
data_dir = Path('input-data')
thp_vol_df = pl.read_csv(data_dir / 'traffic_DLThpVol.csv')
idx_hour_series = thp_vol_df['']
thp_vol_df = thp_vol_df.drop('')

# If in DEBUG mode, shorten dataframes
if DEBUG:
    thp_vol_df = thp_vol_df.head(200).select(thp_vol_df.columns[:800])
    idx_hour_series = idx_hour_series.head(200)

# %% [markdown]
# ## Data Preprocessing

# %%
# Convert idx_hour_series to Polars DataFrame
time_df = pl.DataFrame({'time': idx_hour_series})

# Combine time and thp_vol_df
data_df = pl.concat([time_df, thp_vol_df], how='horizontal')

# %% [markdown]
# ## Initialize W&B

# %%
# Initialize W&B
run = wandb.init(
    project="traffic-forecasting-challenge",
    job_type='train',
    entity="esedx12",
    config=config,
    save_code=True,
    mode=('disabled' if DEBUG else 'online')
)

# %% [markdown]
# ## Linear Regression Model

# %%
# Split into train and test sets based on time
train_percentage = config.get('train_percentage', 0.8)
num_rows = data_df.height
num_train_rows = int(num_rows * train_percentage)

train_df = data_df[:num_train_rows]
test_df = data_df[num_train_rows:]

# Features and targets
X_train_linear = train_df.select(['time'])
Y_train = train_df.drop('time')

X_test_linear = test_df.select(['time'])
Y_test = test_df.drop('time')

# Fit multi-output linear regression using Polars
logging.info("Training the multi-output linear regression model...")

# Fit the model
linear_model = LinearRegression()
linear_model.fit(X_train_linear, Y_train)

# %% [markdown]
# ## Calculate Residuals

# %%
# Predict on training and test sets
logging.info("Calculating residuals...")
Y_train_pred = linear_model.predict(X_train_linear)
Y_test_pred = linear_model.predict(X_test_linear)

# Convert predictions back to Polars DataFrame
Y_train_pred_df = pl.DataFrame(Y_train_pred)
Y_train_pred_df.columns = Y_train.columns

Y_test_pred_df = pl.DataFrame(Y_test_pred)
Y_test_pred_df.columns = Y_test.columns

# Calculate residuals scaled by the linear prediction
# Avoid division by zero by adding a small epsilon
epsilon = 1e-8
train_residuals = (Y_train - Y_train_pred_df) / (Y_train_pred_df + epsilon)
test_residuals = (Y_test - Y_test_pred_df) / (Y_test_pred_df + epsilon)

# %% [markdown]
# ## XGBoost Model for Residuals

# %%
# Prepare feature engineering using existing code
# Assuming utils.create_long_format_df and related functions are available

# Prepare training data for XGBoost
logging.info("Preparing data for XGBoost...")

# We need to prepare features for XGBoost using the existing feature engineering
# Let's assume we have target_dataframes and idx_hour_series ready

# For training data
target_dataframes_train = {
    'residuals': train_residuals,
    # Include any other target dataframes if needed
}
idx_hour_series_train = idx_hour_series[:num_train_rows]

long_train_df = utils.create_long_format_df(
    target_dataframes_train, idx_hour_series_train, config)

X_train_xgb = long_train_df.drop(columns=['residuals'])
y_train_xgb = long_train_df['residuals']

# For test data
target_dataframes_test = {
    'residuals': test_residuals,
}
idx_hour_series_test = idx_hour_series[num_train_rows:]

long_test_df = utils.create_long_format_df(
    target_dataframes_test, idx_hour_series_test, config)

X_test_xgb = long_test_df.drop(columns=['residuals'])
y_test_xgb = long_test_df['residuals']

# Convert to NumPy arrays if necessary
X_train_xgb_np = X_train_xgb.to_numpy()
y_train_xgb_np = y_train_xgb.to_numpy()
X_test_xgb_np = X_test_xgb.to_numpy()
y_test_xgb_np = y_test_xgb.to_numpy()

# %% [markdown]
# ### Fit XGBoost Model

# %%
# Initialize XGBoost regressor
xgb_model = xgb.XGBRegressor(
    **xgb_hyperparams,
    callbacks=[wandb.integration.xgboost.WandbCallback(log_model=True)]
)

# Fit the model
logging.info("Training the XGBoost model on residuals...")
xgb_model.fit(
    X_train_xgb_np,
    y_train_xgb_np,
    eval_set=[(X_train_xgb_np, y_train_xgb_np), (X_test_xgb_np, y_test_xgb_np)],
    verbose=25
)

# %% [markdown]
# ## Combine Predictions

# %%
# Predict residuals on test set using XGBoost
residuals_pred = xgb_model.predict(X_test_xgb_np)

# Scale residuals back up
epsilon = 1e-8
scaled_residuals_pred = residuals_pred * (Y_test_pred_df.to_numpy() + epsilon)

# Final prediction
Y_test_final_pred = Y_test_pred_df.to_numpy() + scaled_residuals_pred

# %% [markdown]
# ## Evaluation

# %%
# Calculate metrics
logging.info("Evaluating the combined model...")
test_mae = mean_absolute_error(Y_test.to_numpy(), Y_test_final_pred, multioutput='raw_values')
test_mse = mean_squared_error(Y_test.to_numpy(), Y_test_final_pred, multioutput='raw_values')

# Calculate average metrics
avg_test_mae = np.mean(test_mae)
avg_test_mse = np.mean(test_mse)

# Log average metrics to W&B
wandb.log({
    'avg_test_mae': avg_test_mae,
    'avg_test_mse': avg_test_mse
})

logging.info(f"Average Test MAE: {avg_test_mae:.4f}")
logging.info(f"Average Test MSE: {avg_test_mse:.4f}")

# %% [markdown]
# ## Save the Models

# %%
# Save the linear model and XGBoost model
model_dir = Path('checkpoints') / wandb.run.name
model_dir.mkdir(parents=True, exist_ok=True)

import pickle
linear_model_path = model_dir / 'linear_model.pkl'
with open(linear_model_path, 'wb') as f:
    pickle.dump(linear_model, f)

xgb_model_path = model_dir / 'xgb_model.ubj'
xgb_model.save_model(str(xgb_model_path))

# Save models to W&B
wandb.save(str(linear_model_path))
wandb.save(str(xgb_model_path))

# %% [markdown]
# ## Plotting Predictions vs Actuals for Sample Beams

# %%
# Sample beams to plot
sample_beams = Y_test.columns[:2]  # Choose two sample beams

for beam in sample_beams:
    # Get actual and predicted values for the beam
    beam_Y_test = Y_test[beam].to_numpy()
    beam_Y_test_pred = Y_test_final_pred[:, Y_test.columns.index(beam)]
    time_test = X_test_linear['time'].to_numpy()

    plt.figure(figsize=(14, 6))
    plt.plot(time_test, beam_Y_test, label='Actual', color='blue')
    plt.plot(time_test, beam_Y_test_pred, label='Predicted', color='orange', linestyle='--')
    plt.title(f'Predictions vs Actuals for Beam: {beam}')
    plt.xlabel('Time')
    plt.ylabel('Throughput Volume')
    plt.legend()
    plt.tight_layout()
    
    # Save plot to W&B
    plot_path = f"beam_{beam}_prediction_plot.png"
    plt.savefig(plot_path)
    wandb.log({f"Prediction Plot for Beam {beam}": wandb.Image(plot_path)})
    plt.show()

# %% [markdown]
# ## Finish W&B Run

# %%
wandb.finish()
