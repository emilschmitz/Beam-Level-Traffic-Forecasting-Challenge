# %% [markdown]
# # XGBoost
# * TODO: add averages for cells and stations, NB inference code will need to be adapted too
# * TODO: target manipulations/engineering
#     * rolling autocorrelation
# * TODO: (vector leaf) multi-output regression
# * TODO: maybe for week 10-11, we should have radically different approach that does not rely on lag feats, since these will be heavily inpacted by compound errors
#     * maybe we should train lin reg for trend and xgboost for seasonality only on idx feats

# %% [markdown]
# ## Roadmap Note
# Regarding your plan to expand the script to first fit a linear model and then apply XGBoost on the residuals, that's a solid approach known as model stacking or residual modeling. This can be set up as a parameter in W&B for flexibility. When you're ready to implement it, you might consider:
# 
# * Implementing a Pipeline: Use scikit-learn's Pipeline to chain the linear model and XGBoost.
# * Parameterization: Add a parameter in your config (e.g., use_linear_model) to toggle this behavior.
# * Logging: Use W&B to track both models' performances separately and combined.

# %%
# %%
from tqdm.notebook import tqdm
from wandb.integration.xgboost import WandbCallback
import wandb
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from optuna_integration.xgboost import XGBoostPruningCallback
from optuna_integration.wandb import WeightsAndBiasesCallback
import optuna
import xgboost as xgb
import sklearn
import numpy as np
import polars as pl
import pandas as pd
import pickle
from typing import Callable
import math
from pathlib import Path
import os
import yaml

import utils

# Manually set the notebook name
# os.environ["WANDB_NOTEBOOK_NAME"] = "xgboost_train.ipynb"

# %%
# %%
def train():
    DEBUG = False
    # config_file_path = Path('configs') / 'linear_config.yaml'
    # config_file_path = Path('configs') / 'autoregressive_config.yaml'
    # config_file_path = Path('configs') / '168hour_shift_config.yaml'
    config_file_path = Path('configs') / 'SWEEP_autoregressive_config.yaml'

    # %%
    # Load the YAML configuration file
    # with open(config_file_path, 'r') as file:
        # config = yaml.safe_load(file)

    # Extract xgb_hyperparams from config
    # xgb_hyperparams = config.get('xgb_hyperparams', {})

    # Merge xgb_hyperparams into config
    # config.update(xgb_hyperparams)

    # %%
    # Initialize W&B
    run = wandb.init(
        project="traffic-forecasting-challenge",
        job_type='train',
        entity="esedx12",
        save_code=True,
        mode=('dryrun' if DEBUG else 'online')
    )
    # %%
    # Read the CSV files
    data_dir = Path('input-data')
    target_dataframes = {
        # This is the target variable
        'thp_vol': pl.read_csv(data_dir / 'traffic_DLThpVol.csv'),
        'prb': pl.read_csv(data_dir / 'traffic_DLPRB.csv'),
        'thp_time': pl.read_csv(data_dir / 'traffic_DLThpTime.csv'),
        'mr_number': pl.read_csv(data_dir / 'traffic_MR_number.csv')
    }

    # Filter target dataframes based on config
    target_dataframes = {
        k: v for k, v in target_dataframes.items() if k in wandb.config.target_df_names}

    idx_hour_series = target_dataframes['thp_vol']['']

    # Drop the first column (idx hour) from each dataframe
    for k in target_dataframes:
        target_dataframes[k] = target_dataframes[k].drop('')

    # Debug mode: shorten dataframes and config lists
    if DEBUG:
        target_dataframes = {k: v.head(200).select(
            v.columns[:800]) for k, v in target_dataframes.items()}
        config = {k: v[:3] if isinstance(
            v, list) else v for k, v in config.items()}

    # Merge xgb_hyperparams into config
    # config.update(xgb_hyperparams)

    # %%
    # %%

    # Save utils.py to W&B
    utils_path = Path('utils.py')
    if utils_path.exists():
        wandb.save(str(utils_path))

    # %% [markdown]
    #  ## Feature Engineering
    # 
    #  The feature engineering steps are handled by utility functions.

    # %%
    # %%
    # Use first config.train_percentage of dataframe rows for training, and the rest for validation and testing
    num_rows = len(target_dataframes['thp_vol'])
    num_train_rows = round(num_rows * wandb.config.train_percentage)
    num_val_rows = round(num_rows * wandb.config.val_percentage)

    config = wandb.config.as_dict()

    # Make feature dataframes
    feature_dfs = utils.create_all_feature_dfs(
        target_dataframes, idx_hour_series, config)

    train_target_dfs = {k: v.head(num_train_rows)
                        for k, v in target_dataframes.items()}
    train_feature_dfs = {k: v.head(num_train_rows)
                        for k, v in feature_dfs.items()}
    train_idx_hour_series = idx_hour_series.head(num_train_rows)

    val_target_dfs = {k: v.slice(num_train_rows + 1, num_val_rows)
                    for k, v in target_dataframes.items()}
    val_feature_dfs = {k: v.slice(num_train_rows + 1, num_val_rows)
                    for k, v in feature_dfs.items()}
    val_idx_hour_series = idx_hour_series.slice(num_train_rows + 1, num_val_rows)

    # %%
    # Create long format dataframes using utility functions
    long_train_df = utils.create_long_format_df(
        train_target_dfs, train_feature_dfs, train_idx_hour_series, wandb.config)
    long_val_df = utils.create_long_format_df(
        val_target_dfs, val_feature_dfs, val_idx_hour_series, wandb.config)

    target_cols = list(target_dataframes.keys())

    # Assuming long_train_df and long_val_df are pandas DataFrames
    X_train = long_train_df.drop(columns=target_cols)
    y_train = long_train_df[target_cols]

    X_val = long_val_df.drop(columns=target_cols)
    y_val = long_val_df[target_cols]

    wandb.config.update({
        'num_train_samples': len(X_train),
        'num_val_samples': len(X_val),
        'features': X_train.columns.to_list(),
        'targets': y_train.columns.to_list()
    })

    # %% [markdown]
    #  ## Train Models
    # *  TODO if indicated for performance reasons, get the max idx_hour with a null and return it so we can shorten the df for multi-step predict
    # * TODO also add target transformations (maybe sklearn can help)
    # * TODO normalize somehow if data is on very different scales for different beams

    # %% [markdown]
    # ### Fit models

    # %%
    # sk-learn linear model
    if wandb.config.model == 'linear':
        models = {}
        for target in target_cols:
            model = sklearn.linear_model.LinearRegression()
            model.fit(pd.get_dummies(X_train), y_train[target])
            models[target] = model
            # wandb log and print some metrics, like mae
            y_pred = model.predict(pd.get_dummies(X_val))
            mae = sklearn.metrics.mean_absolute_error(y_val[target], y_pred)
            wandb.log({f'mae_{target}': mae})
            print(f'MAE for {target}: {mae}')

    # %%
    X_train.columns[:10]

   
    # %%
    # xgboost model
    if wandb.config.model == 'xgboost':
        models = {}
        for target_name in y_train.columns[:1]:
            model = xgb.XGBRegressor(
                **config, callbacks=[WandbCallback(log_model=True)])
            print(f"\nFitting model for {target_name}:")
            model.fit(
                X_train,
                y_train[target_name],
                eval_set=[(X_train, y_train[target_name]),
                        (X_val, y_val[target_name])],
                verbose=25
            )
            models[target_name] = model

    # %% [markdown]
    # ### Save models

    # %%
    # %%
    for target_name, model in models.items():
        model_dir = Path('checkpoints') / wandb.run.name
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f'{target_name}.ubj'
        pickle.dump(model, open(model_path, 'wb'))
        wandb.save(str(model_path))
    # %%
    wandb.finish()

if __name__ == '__main__':
    train()
# %%
# %%

# %%



