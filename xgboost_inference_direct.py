# %% [markdown]
# # Inference

# %%
from pathlib import Path
import os
os.environ["WANDB_NOTEBOOK_NAME"] = "xgboost_inference.ipynb"  # Manually set the notebook name

import pandas as pd
import polars as pl
import xgboost as xgb
import wandb
from tqdm.notebook import tqdm
import pickle
import numpy as np

import utils
import yaml

# %%
DEBUG = True
# Load the inference config from the YAML file

with open('configs/direct_inference_config_11_10_24.yaml', 'r') as f:
    train_config = yaml.safe_load(f)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

train_config = dotdict(train_config)

inference_config = {
    'prediction_length': 168,
    'create_submission_csv': False
}

# Update the checkpoints directory to 'checkpoints_final'
checkpoints_dir = 'checkpoints_final'
xgboost_models_dir = Path(checkpoints_dir)

# Load all models
models = {}
for file_name in os.listdir(xgboost_models_dir):
    if file_name.startswith('forward_shift_'):
        shift = int(file_name.split('_')[-1])
        model_path = xgboost_models_dir / file_name
        with open(model_path, 'rb') as f:
            models[shift] = pickle.load(f)

# %% [markdown]
# ## Load and Prepare Data

# %%
# Read the CSV files
data_dir = Path('input-data')
target_dataframes = {
    'thp_vol': pl.read_csv(data_dir / 'traffic_DLThpVol.csv'),  # This is the target variable
    'prb': pl.read_csv(data_dir / 'traffic_DLPRB.csv'),
    'thp_time': pl.read_csv(data_dir / 'traffic_DLThpTime.csv'),
    'mr_number': pl.read_csv(data_dir / 'traffic_MR_number.csv')
}

template_df = target_dataframes['thp_vol']
idx_hour_series = template_df['idx_hour']

predict_hour = 840

null_row = pl.DataFrame({beam_id: [None] for beam_id in template_df.columns})
target_dataframes = {k: pl.concat([v, null_row], how='vertical_relaxed') for k, v in target_dataframes.items()}

target_names = list(target_dataframes.keys())
feature_dfs = utils.create_all_feature_dfs(target_dataframes, idx_hour_series, train_config)
feature_dfs = {k: v.tail(1) for k, v in feature_dfs.items()}  # maybe turn in to lazyframe for efficiency?
X_predict = utils.convert_to_long_format(feature_dfs)

cat_types = utils.make_id_cat_type(template_df.columns)
X_predict = X_predict.to_pandas()
for col in ['beam_id', 'cell_id', 'station_id']:
    if col in X_predict.columns:
        X_predict[col] = X_predict[col].astype(cat_types[col])
# %%

beam_id_col = pl.DataFrame({beam_id: [beam_id] for beam_id in template_df.columns})
ys_predicted = []
for shift in range(168):
    y_predicted = models[shift].predict(X_predict)

    idx_hour = pl.DataFrame({'idx_hour': [840 + shift] * len(template_df.columns)})

    y_predicted_long_df = pl.concat([beam_id_col, idx_hour, pl.DataFrame(y_predicted)], how='horizontal')
    y_predict_wide = utils.convert_to_wide_format(y_predicted_long_df, output_df_names=['thp_vol'])
    ys_predicted = ys_predicted.append(y_predicted)

# %%
predictions_wide = pl.concat(ys_predicted, how='vertical')
predictions_wide = predictions_wide.with_columns(idx_hour=range(840, 1008))

# %%

# We need these long-format columns to convert the predictions to wide format
util_dfs = {}
util_dfs['idx_hour'] = pl.DataFrame({beam_id: [predict_hour] for beam_id in template_df.columns})
util_long_df = utils.convert_to_long_format(util_dfs)
ys_predicted_long = pl.concat([util_long_df, ys_predicted_long], how='horizontal')

y_predicted_wide = utils.convert_to_wide_format(ys_predicted_long, output_df_names=target_names)    