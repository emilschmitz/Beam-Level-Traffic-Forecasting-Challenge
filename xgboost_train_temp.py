# %% [markdown]
# # XGBoost
# * TODO: add averages for cells and stations, NB inference code will need to be adapted too
# * TODO: target manipulations/engineering
#     * rolling autocorrelation
# * TODO: (vector leaf) multi-output regression
# * TODO: maybe for week 10-11, we should have radically different approach that does not rely on lag feats, since these will be heavily inpacted by compound errors
#     * maybe we should train lin reg for trend and xgboost for seasonality only on idx feats

# %%
import math
from pathlib import Path
import os
os.environ["WANDB_NOTEBOOK_NAME"] = "xgboost_train.ipynb"  # Manually set the notebook name
from typing import Callable

import polars as pl
import numpy as np
import xgboost as xgb
import optuna
from optuna_integration.wandb import WeightsAndBiasesCallback
from optuna_integration.xgboost import XGBoostPruningCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.compose import TransformedTargetRegressor
import shap

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

import wandb
from wandb.integration.xgboost import WandbCallback
from tqdm.notebook import tqdm

# %%
DEBUG = False

# %%
# Read the CSV files
data_dir = Path('input-data')
thp_vol = pl.read_csv(data_dir / 'traffic_DLThpVol.csv')  # This is the target variable
prb = pl.read_csv(data_dir / 'traffic_DLPRB.csv')
thp_time = pl.read_csv(data_dir / 'traffic_DLThpTime.csv')
mr_number = pl.read_csv(data_dir / 'traffic_MR_number.csv')

target_dataframes = {
    'thp_vol': thp_vol,
    'prb': prb,
    'thp_time': thp_time,
    'mr_number': mr_number
}

# Rename first col to 'hour'
for k, v in target_dataframes.items():
    target_dataframes[k] = v.rename({'': "idx_hour"})

# %%
xgb_hyperparams = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    # 'max_depth': 6,
    'eta': 0.1,
    'subsample': 0.7,
    # 'colsample_bytree': 0.8,
    # 'verbosity': 2,
    'early_stopping_rounds': 10,
    'n_estimators': 100,
}

config = {
    'lags': [1, 2, 3, 6, 12, 13, 14, 24, 25, 26, 48, 49, 72],
    'rolling_avgs': [1, 3, 9, 24, 48, 72, 86],
    'delta_reference_points': [(1, 2), (1, 3), (1, 6), (1, 24), (24, 25), (48, 49)],
    'std_windows': [3, 6, 12, 24, 48, 72, 86],
    'num_zeros_windows': [6, 12, 24],
    'hour_shifts': [0, 6, 12, 18],
    'weekday_shifts': [0, 3, 6],
    'train_percentage': 0.6,
    'val_percentage': 0.3,  # The rest is test
    'run_shap': False,
    'target_df_names': [  # dataframes used as target variables
        'thp_vol', 
        'mr_number',
        # 'vol_per_prb',
    ],
    'feat_base_df_names': [  # dataframes used to create the features
        'thp_vol', 
        'mr_number',
        'vol_per_user', 
        # 'vol_per_prb',
    ],
}

if DEBUG:
    target_dataframes = {k: v.head(400).select(v.columns[:800]) for k, v in target_dataframes.items()}
    # shorten every list in config to max three elements
    config = {k: v[:3] if isinstance(v, list) else v for k, v in config.items()}

config = xgb_hyperparams | config

# %%
run = wandb.init(project="traffic-forecasting-challenge", job_type='train', entity="esedx12", config=config, save_code=True, mode=('dryrun' if DEBUG else 'online'))

# %% [markdown]
# ## Create interactions between targets

# %%
def make_interaction_dataframe(name, target_dataframes, eps=0.1):
    match name:
        case 'vol_per_user':
            return target_dataframes['thp_vol'] / (target_dataframes['mr_number'] + eps)
        case 'vol_per_prb':
            return target_dataframes['thp_vol'] / (target_dataframes['prb'] + eps)
        case 'vol_per_time':
            return target_dataframes['thp_vol'] / (target_dataframes['thp_time'] + eps)
        case 'prb_per_user':
            return target_dataframes['prb'] / (target_dataframes['mr_number'] + eps)
        case 'prb_per_time':
            return target_dataframes['prb'] / (target_dataframes['thp_time'] + eps)

# %%
def make_base_dataframes(target_dataframes: dict[str, pl.DataFrame], base_df_names: list[str]) -> dict[str, pl.DataFrame]:
    """
    Create new dataframes to be used for feature eng or as targets.
    """
    base_dataframes = {}
    for name in base_df_names:
        try:
            base_dataframes[name] = target_dataframes[name]
        except KeyError:
            base_dataframes[name] = make_interaction_dataframe(name, target_dataframes)

    return base_dataframes

# %% [markdown]
# ### **Modify targets** based on `config['target_df_names']`

# %%
target_dataframes = make_base_dataframes(target_dataframes, config['target_df_names'])

# %% [markdown]
# ## Feature Engineering

# %%
# TODO if indicated for performance reasons, get the max idx_hour with a null and return it so we can shorten the df for multi-step predict
# TODO also add target transformations (maybe sklearn can help)
# TODO normalize somehow if data is on very different scales for different beams

# %% [markdown]
# ### Hours of day, days of week

# %%
def create_hour_feats(idx_hour_df: pl.DataFrame, lags: list[int]) -> dict[str, pl.DataFrame]:
    """
    Create shifted versions of daily 24h count for each column in the DataFrame.
    Returns a dictionary of DataFrames with the key format '24h_shifted_{lag}h'.
    """
    feature_dfs = {}
    daily_hour_df = idx_hour_df % 24  # Calculate daily hours

    for lag in lags:
        shifted_df = (daily_hour_df + lag) % 24
        feature_dfs.update({f'daily_hours_shifted_{lag}h': shifted_df})

    return feature_dfs

def create_weekday_feats(idx_hour_df: pl.DataFrame, lags: list[int]) -> dict[str, pl.DataFrame]:
    """
    Create shifted versions of weekday count for each column in the DataFrame.
    Returns a dictionary of DataFrames with the key format 'weekday_shifted_{lag}d'.
    """
    feature_dfs = {}
    day_df = idx_hour_df // 24  # Calculate days
    weekday_df = day_df % 7  # Calculate weekdays

    for lag in lags:
        shifted_df = (weekday_df + lag) % 7
        feature_dfs.update({f'weekday_shifted_{lag}d': shifted_df})

    return feature_dfs

# %%
def create_time_feature_dfs(idx_hour_df: pl.DataFrame, idx_hour_shifts: int, weekday_shifts: int) -> dict[str, pl.DataFrame]:
    """
    Create repeating 24h and 7d features.
    """
    hour_feats = create_hour_feats(idx_hour_df, idx_hour_shifts)
    weekday_feats = create_weekday_feats(idx_hour_df, weekday_shifts)

    return {**hour_feats, **weekday_feats}

# %% [markdown]
# ### Time-series features

# %%
def create_ts_feature_dfs(df_name: str, df: pl.DataFrame, lags: list[int], rolling_avgs: list[int], delta_reference_points: list[tuple[int, int]], std_windows: list[int], num_zeros_windows: list[int]) -> dict[str, pl.DataFrame]:
    """
    Create lag, rolling average, delta, and standard deviation features for all columns in the DataFrame.
    Returns a dict of DataFrames.
    """
    lag_feats = {f"{df_name}_lag_{lag}": df.shift(lag) for lag in lags}

    delta_feats = {f"{df_name}_delta_{point_pair[0]}_{point_pair[1]}": (df.shift(point_pair[0]) - df.shift(point_pair[1])) for point_pair in delta_reference_points}

    # We need to shift one step to aviod data leakage
    shifted_df = df.shift(1)
    rolling_avg_feats = {
        f"{df_name}_rolling_avg_{window}":
            pl.DataFrame({
                col: shifted_df[col].rolling_mean(window_size=window)
                for col in df.columns
            })
        for window in rolling_avgs
    }

    std_feats = {
        f"{df_name}_std_{window}":
            pl.DataFrame({
                col: shifted_df[col].rolling_std(window_size=window)
                for col in df.columns
            })
        for window in std_windows
    }

    num_zeros_feats = {
        f"{df_name}_rolling_avg_{window}":
            pl.DataFrame({
                col: (shifted_df[col] == 0).cast(pl.Float32).rolling_sum(window_size=window)
                for col in df.columns
            })
        for window in rolling_avgs
    }

    return {**lag_feats, **rolling_avg_feats, **delta_feats, **std_feats, **num_zeros_feats}

# %% [markdown]
# ### Aggregations over cells, stations

# %%
def extract_cell_id(beam_id: str) -> str:
    """
    Extract the cell ID from a beam ID.
    """
    return "_".join(beam_id.split("_")[0:2])

def extract_station_id(beam_id: str) -> int:
    """
    Extract the station ID from a beam ID.
    """
    return beam_id.split("_")[0]

def compute_cell_avg(dataframe: pl.DataFrame, cell_id: str) -> pl.Series:
    """
    Compute the average of a DataFrame for a given cell.
    """
    cell_cols = [col for col in dataframe.columns if col.startswith(cell_id)]
    cell_data = dataframe.select(cell_cols)
    return cell_data.mean(axis=1)

def compute_station_avg(dataframe: pl.DataFrame, station_id: int) -> pl.Series:
    """
    Compute the average of a DataFrame for a given station.
    """
    station_cols = [col for col in dataframe.columns if col.startswith(station_id)]
    station_data = dataframe.select(station_cols)
    return station_data.mean(axis=1)

# %%
# def create_id_features(beam_id: str, length: int) -> pl.DataFrame:
#     """
#     Create DataFrame columns for cell and station IDs based on beam ID.
#     """
#     beam_id_col = pl.Series([beam_id] * length).alias("beam_id").to_frame()

#     cell_id = extract_cell_id(beam_id)
#     cell_id_col = pl.Series([cell_id] * length).alias("cell_id").to_frame()

#     station_id = extract_station_id(beam_id)
#     station_id_col = pl.Series([station_id] * length).alias("station_id").to_frame() 

#     return pl.concat([beam_id_col, cell_id_col, station_id_col], how="horizontal")

# %%
def create_id_feature_dfs(template_df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """
    Create DataFrames for beam, cell and station IDs based on a template DataFrame.
    """
    id_feature_dfs = {}
    id_feature_dfs['beam_id'] = pl.DataFrame({beam_id: [beam_id] * len(template_df) for beam_id in template_df.columns})
    id_feature_dfs['cell_id'] = pl.DataFrame({beam_id: [extract_cell_id(beam_id)] * len(template_df) for beam_id in template_df.columns})
    id_feature_dfs['station_id'] = pl.DataFrame({beam_id: [extract_station_id(beam_id)] * len(template_df) for beam_id in template_df.columns})

    return id_feature_dfs

# %% [markdown]
# ### Combine all the features

# %%
def create_all_feature_dfs(target_dataframes: dict[str, pl.DataFrame], config: wandb.Config) -> dict[str, pl.DataFrame]:
    """
    Create features for the traffic forecasting model for all beams at once.
    Returns a dictionary of feature DataFrames.
    """
    feature_dfs = {}
    template_df = target_dataframes['thp_vol']

    # Create a DataFrame full of the idx_hour series repeated across all columns
    idx_hour_series = template_df['idx_hour']
    beam_ids = template_df.drop('idx_hour').columns
    feature_dfs['idx_hour'] = pl.DataFrame(
        {beam_id: idx_hour_series for beam_id in beam_ids})
    feature_dfs['beam_id'] = pl.DataFrame(
        {beam_id: [beam_id] * len(template_df) for beam_id in beam_ids})

    # Repeating 24h and 7d features
    feature_dfs.update(create_time_feature_dfs(
        feature_dfs['idx_hour'], config.hour_shifts, config.weekday_shifts))
    
    # Make list of dataframes, on which ts features will be created
    base_dataframes = make_base_dataframes(target_dataframes, config.feat_base_df_names)

    for df_name, df in base_dataframes.items():
        df = df.drop('idx_hour')

        feature_dfs.update(create_ts_feature_dfs(df_name, df, config.lags, config.rolling_avgs,
                           config.delta_reference_points, config.std_windows, config.num_zeros_windows))

    return feature_dfs

# %% [markdown]
# ## Data Formatting

# %%
def convert_to_long_format(dataframes: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """
    Convert the target and feature DataFrames to a long/tidy format.
    Drop nulls.
    """
    columnized = []
    for df_name, df in dataframes.items():
        columnized.append(df.unpivot(value_name=df_name).select(df_name))

    return pl.concat(columnized, how='horizontal').drop_nulls()


def convert_to_wide_format(dataframe: pl.DataFrame, output_df_names: list[str]) -> dict[str, pl.DataFrame]:
    """
    Convert the target and feature DataFrames to a dict of wide format DataFrames.
    """
    wide_dfs = {}
    for df_name in output_df_names:
        wide_df = dataframe.pivot(index='idx_hour', columns='beam_id', values=df_name)
        wide_dfs[df_name] = wide_df

    return wide_dfs

# %%
def create_long_train_df(target_dataframes: dict[str, pl.DataFrame], config: wandb.Config) -> pl.DataFrame:
    """
    Create long DataFrame with features and target cols
    """
    feature_dfs = create_all_feature_dfs(target_dataframes, config)
    target_dfs = {k: v.drop('idx_hour') for k, v in target_dataframes.items()}
    all_train_dfs = {**feature_dfs, **target_dfs}
    long_train_df = convert_to_long_format(all_train_dfs)
    return long_train_df

# %%
# Use first config.train_percentage of dataframe rows for training, and the rest for validation and testing
num_rows = len(target_dataframes['thp_vol'])
num_train_rows = round(num_rows * wandb.config.train_percentage)
num_val_rows = round(num_rows * wandb.config.val_percentage)

train_dataframes = {k: v.head(num_train_rows) for k, v in target_dataframes.items()}
val_dataframes = {k: v.slice(num_train_rows + 1, num_train_rows + num_val_rows) for k, v in target_dataframes.items()}

# %%
long_train_df = create_long_train_df(target_dataframes, wandb.config)
long_val_df = create_long_train_df(val_dataframes, wandb.config)

dropped_cols = ['idx_hour', 'beam_id']
target_cols = list(target_dataframes.keys())

X_train, y_train = long_train_df.drop(dropped_cols + target_cols), long_train_df.select(target_cols)
X_val, y_val = long_val_df.drop(dropped_cols + target_cols), long_val_df.select(target_cols)

wandb.config.update({'train_shape': X_train.shape, 'val_shape': X_val.shape, 'features': X_train.columns, 'targets': y_train.columns})

# %% [markdown]
# ## Train
# * We use the Scikit-Learn API
# * TODO add optuna
# * wandbc = WeightsAndBiasesCallback(metric_name="accuracy", wandb_kwargs=wandb_kwargs

# %%
models = {}
for target_name in y_train.columns:
    model = xgb.XGBRegressor(**xgb_hyperparams, callbacks=[WandbCallback(log_model=True)])
    print(f"\nFitting model for {target_name}:")
    model.fit(X_train, y_train[target_name], eval_set=[(X_train, y_train[target_name]), (X_val, y_val[target_name])], verbose=25)
    models[target_name] = model

# %%
for target_name, model in models.items():
    model_dir = Path('checkpoints') / wandb.run.name
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f'{target_name}.ubj'
    model.save_model(model_path)

# %%
wandb.run.dir

# %% [markdown]
# ## Charts, Tables

# %%
# Iterate through each model in model.models
for target, target_model in models.items():
    print(f"Processing target: {target}")

    # Predict
    train_preds = target_model.predict(X_train.to_numpy())
    val_preds = target_model.predict(X_val.to_numpy())

    # Compute MAE values
    train_mae = mean_absolute_error(y_train[target].to_numpy(), train_preds)
    val_mae = mean_absolute_error(y_val[target].to_numpy(), val_preds)

    # Log the best score to wandb
    best_iteration = target_model.best_iteration
    best_val_mae = target_model.evals_result()['validation_1']['mae'][best_iteration]
    best_train_mae = target_model.evals_result()['validation_0']['mae'][best_iteration]

    wandb.log({
        f'{target}_best_val_mae': best_val_mae, 
        f'{target}_best_round': best_iteration, 
        f'{target}_best_train_mae': best_train_mae
    })

    # Convert evaluation results to a DataFrame
    evals_result = target_model.evals_result()
    rounds = range(1, len(evals_result['validation_0']['mae']) + 1)

    # Create a DataFrame using polars
    eval_df = pl.DataFrame({
        'Round': rounds,
        'Train MAE': evals_result['validation_0']['mae'],
        'Val MAE': evals_result['validation_1']['mae']
    })

    # Log eval_df to wandb
    wandb.log({f'{target}_eval_df': wandb.Table(data=eval_df.to_pandas())})

    # Plot the results using Plotly
    fig = px.line(
        x=rounds, 
        y=[evals_result['validation_0']['mae'], evals_result['validation_1']['mae']],
        labels={'x': 'Boosting Round', 'value': 'Mean Absolute Error'}, 
        title=f'Training and Val MAE over Boosting Rounds for {target}'
    )
    fig.update_layout(
        legend=dict(
            title='Legend',
            itemsizing='constant'
        ),
        legend_title_text='Dataset'
    )
    fig.data[0].name = 'Train MAE'
    fig.data[1].name = 'Val MAE'

    # Log the plot to wandb
    wandb.log({f"{target}_MAE_Plot": fig})

    # Optionally, display the plot
    fig.show()

    print(f"Best Val MAE for {target}: {best_val_mae}")
    print(f"Round: {best_iteration}")

# %%
if wandb.config.run_shap:
    # Create a SHAP explainer for the XGBoost model
    explainer = shap.TreeExplainer(model.models['thp_vol'], X_val.to_pandas())

    # Calculate SHAP values for the val set
    explanation = explainer(X_val.to_pandas())

    # Upload plots and SHAP values to wandb
    wandb.log({"SHAP Bar Plot": shap.plots.bar(explanation, max_display=30)})
    wandb.log({"SHAP Summary Plot": shap.summary_plot(explanation, X_val.to_pandas())})

    # # Optional: Generate a dependence plot for a specific feature (replace 'feature_index' with the actual feature index)
    # shap.dependence_plot(0, shap_values, X_val.to_pandas())

    # # Optional: Generate a force plot for the first instance in the val set
    # shap.force_plot(explainer.expected_value, shap_values[0, :], X_val.to_pandas()[0, :])

# %%
wandb.finish()

# %%



