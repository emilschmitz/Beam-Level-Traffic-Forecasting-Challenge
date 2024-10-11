# %% [markdown]
# # Inference

# %%
from pathlib import Path
import os
os.environ["WANDB_NOTEBOOK_NAME"] = "xgboost_inference.ipynb"  # Manually set the notebook name

import polars as pl
import xgboost as xgb
import wandb
from tqdm.notebook import tqdm
import pickle

import utils

# %%
DEBUG = True

# %%
# Choose training run from which to load the model, etc.
train_run_name = 'prime-wind-87'
run_path = f'esedx12/traffic-forecasting-challenge/{train_run_name}'
api = wandb.Api()
train_run = api.runs(
    path="esedx12/traffic-forecasting-challenge",
    filters={"display_name": {"$eq": train_run_name}} 
)[0]
train_config = train_run.config

# %%
inference_config = {
    # 'prediction_start': train_config['train_shape'][0] + 1,
    'prediction_length': 1900 - train_config['num_train_rows'] if not DEBUG else 5,
}

# %%
run = wandb.init(project="traffic-forecasting-challenge", tags=[train_run_name], job_type='inference',
                 entity="esedx12", config=inference_config, save_code=True, mode=('dryrun' if DEBUG else 'online'))

# %%
checkpoints_dir = 'checkpoints'
xgboost_models_dir = Path(checkpoints_dir) / train_run_name

models = {}
for file_name in os.listdir(xgboost_models_dir):
    if file_name.endswith('.ubj'):
        target_name = file_name[:-4]
        model_path = xgboost_models_dir / file_name
        models[target_name] = pickle.load(open(model_path, 'rb'))

# %% [markdown]
# ## Load and prepare data

# %%
# %%
# Read the CSV files
data_dir = Path('input-data')
target_dataframes = {
    'thp_vol': pl.read_csv(data_dir / 'traffic_DLThpVol.csv'),  # This is the target variable
    'prb': pl.read_csv(data_dir / 'traffic_DLPRB.csv'),
    'thp_time': pl.read_csv(data_dir / 'traffic_DLThpTime.csv'),
    'mr_number': pl.read_csv(data_dir / 'traffic_MR_number.csv')
}

# Filter target dataframes based on train_config
target_dataframes = {k: v for k, v in target_dataframes.items() if k in train_config['target_df_names']}

idx_hour_series = target_dataframes['thp_vol']['']

# Drop the first column (idx hour) from each dataframe
for k in target_dataframes:
    target_dataframes[k] = target_dataframes[k].rename({'': 'idx_hour'})

# A long format beam_id column to be used for converting to wide format
beam_id_col = utils.convert_to_long_format({'beam_id': pl.DataFrame({beam_id: [beam_id] * len(target_dataframes['thp_vol']) for beam_id in target_dataframes['thp_vol'].columns})})

# %%
num_rows = len(target_dataframes['thp_vol'])
num_train_rows = round(num_rows * train_config['train_percentage'])
# num_val_rows = round(num_rows * train_config['val_percentage'])

# Split data into train and test
input_dataframes = {k: v.drop('idx_hour').head(num_train_rows) for k, v in target_dataframes.items()}
input_idx_hour_series = idx_hour_series.head(num_train_rows)

comparison_dataframes = {k: v.slice(num_train_rows, inference_config['prediction_length']) for k, v in target_dataframes.items()}
# TODO add different df sets form idx of validation and holdout test

# %% [markdown]
# ## Multi-Step Inference

# %%
def predict_one_step(target_dataframes: dict[pl.DataFrame], idx_hour_series: pl.Series ,models: xgb.Booster, train_config: wandb.Config) -> dict[pl.DataFrame]:
    """
    Predict one step into the future using a trained model.
    Takes DataFrames of len n, returns DataFrames of len n + 1.
    """
    template_df = target_dataframes['thp_vol']
    predict_hour = idx_hour_series[-1] + 1

    null_row = pl.DataFrame({beam_id: [None] for beam_id in template_df.columns})
    target_dataframes = {k: pl.concat([v, null_row], how='vertical_relaxed') for k, v in target_dataframes.items()}

    target_names = list(target_dataframes.keys())
    feature_dfs = utils.create_all_feature_dfs(target_dataframes, idx_hour_series, train_config)
    feature_dfs = {k: v.tail(1) for k, v in feature_dfs.items()}  # maybe turn in to lazyframe for efficiency?
    X_predict = utils.convert_to_long_format(feature_dfs)

    # We predict only the idx immediately folling the last idx in the input, ie a single row
    ys_predicted_long = pl.DataFrame()
    for target_name, model in models.items():
        y_predicted = model.predict(X_predict.to_numpy())
        ys_predicted_long = pl.concat([ys_predicted_long, pl.DataFrame({target_name: y_predicted})], how='horizontal')

    # We need these long-format columns to convert the predictions to wide format
    util_dfs = {}
    util_dfs['beam_id'] = pl.DataFrame({beam_id: [beam_id] for beam_id in template_df.columns})
    util_dfs['idx_hour'] = pl.DataFrame({beam_id: [predict_hour] for beam_id in template_df.columns})
    util_long_df = utils.convert_to_long_format(util_dfs)
    ys_predicted_long = pl.concat([util_long_df, ys_predicted_long], how='horizontal')

    y_predicted_wide = utils.convert_to_wide_format(ys_predicted_long, output_df_names=target_names)    

    return (
        {target_name: pl.concat([target_dataframes[target_name].head(-1), y_predicted_wide[target_name]], how='vertical_relaxed') for target_name in target_names},
        idx_hour_series.append(pl.Series([predict_hour]))
        )

# %%
def predict_multi_step(target_dataframes: dict[pl.DataFrame], idx_hour_series: pl.Series, models: xgb.Booster, train_config: wandb.Config, num_steps: int, max_lag=None) -> dict[pl.DataFrame]:
    """
    Predict multiple steps into the future using a trained model.
    Takes DataFrames of len n, returns DataFrames of len n + num_steps.
    
    Args:
        target_dataframes (dict): A dictionary of DataFrames representing the target data.
        idx_hour_series (Series): Index hours CORRESPONDING to target_dataframes.

    Returns:
        dict: A dictionary of DataFrames representing the predicted target dataframes.
    """
    if max_lag:
        target_dataframes = {k: v.tail(max_lag + 5) for k, v in target_dataframes.items()}
        idx_hour_series = idx_hour_series.tail(max_lag + 5)

    for _ in tqdm(range(num_steps), desc='Predicting steps...'):
        target_dataframes, idx_hour_series = predict_one_step(target_dataframes, idx_hour_series, models, train_config)

    return {k: pl.concat([pl.DataFrame({'idx_hour': idx_hour_series}), v], how='horizontal') for k, v in target_dataframes.items()}

# %%
ys_pred = predict_multi_step(input_dataframes, input_idx_hour_series, models, train_config=train_config, num_steps=inference_config['prediction_length'])

# %%
def mean_absolute_error(Y_true: pl.DataFrame, Y_pred: pl.DataFrame) -> float:
    """
    Compute the mean absolute error between two DataFrames.
    """
    # TODO some kind of check here even though idx_hour is no longer normally part of dfs
    assert (Y_true['idx_hour'] == Y_pred['idx_hour']).all(), "DataFrames must be aligned"
    # assert Y_true.shape == Y_pred.shape, "DataFrames must have the same shape"

    return (Y_true - Y_pred).select(pl.all().abs().mean()).mean_horizontal()[0]

# %%
comparison_dataframes['thp_vol'] 
ys_pred['thp_vol'].slice(503, 515).head(15)

# %%
mean_absolute_error(comparison_dataframes['thp_vol'], ys_pred['thp_vol'].tail(inference_config['prediction_length']))

# %% [markdown]
# ## ...on Validation Set

# %% [markdown]
# ## ...on Test Set

# %% [markdown]
# ## ...on Validation and Test Sets

# %% [markdown]
# ## Create Submission CSV
# 
# * Hours in 5 weeks: 840
# * Hours in 6 weeks: 1008
# * We need period 841-1008 (841:1009 with Python list indexing)
# 
# * Hours in 10 weeks: 1680
# * Hours in 11 weeks: 1848

# %%
def create_half_submission_df(input_df: pl.DataFrame, weeks: str) -> pl.DataFrame:
    """
    Create a submission CSV file from a Polars DataFrame of thp_vol.
    """
    if weeks == '5w-6w':
        range = [841, 1008]
    elif weeks == '10w-11w':
        range = [1681, 1848]

    # Choose rows with first column 'idx_hour' having the values 671-840.
    input_df = input_df.filter(pl.col('idx_hour') >= range[0], pl.col('idx_hour') <= range[1])

    # Some checks on the input_df
    assert input_df.shape == (168, 2881), f"Expected shape (168, 2881), got {input_df.shape}"
    assert input_df.select(pl.any_horizontal(pl.all().is_null().any())).item() == False, "Submission dataframe contains null values"
    assert input_df['idx_hour'].head(1)[0] <= range[0] and input_df['idx_hour'].tail(1)[0] >= range[1], "Submission dataframe does seemingly not contain the correct idx_hour values"

    # Stack the dataframe with f'traffic_DLThpVol_test_5w-6w_{hour}_{beam_id}' as index
    # where it cycles through the values 671-840 for hour and then the beam_ids, which are colnames of input_df
    # return input_df.unpivot(index='idx_hour')
    return input_df.unpivot(index='idx_hour', variable_name='beam_id').with_columns(
        pl.concat_str([pl.lit('traffic_DLThpVol_test'), pl.lit(weeks), pl.col('idx_hour') - range[0], pl.col('beam_id')], separator='_').alias('ID')
    ).select(['ID', 'value']).rename({'value': 'Target'})


def create_submission_csv(input_df: pl.DataFrame, output_filename='traffic_forecast.csv', archiving_dir='submission-csvs-archive') -> pl.DataFrame:
    """
    Create a submission CSV file from data in input format that's been extended to cover weeks 5-6 and 10-11.
    """

    # Create half submission dataframes
    half_submission_5w_6w = create_half_submission_df(input_df, '5w-6w')
    half_submission_10w_11w = create_half_submission_df(input_df, '10w-11w')

    # Concatenate the two half submission dataframes
    submission_df = pl.concat([half_submission_5w_6w, half_submission_10w_11w], how='vertical')

    # Save the submission dataframe to a CSV file for submission, and to wandb
    submission_df.write_csv(output_filename)
    wandb.save(output_filename)

    # Save the submission dataframe to a CSV file for archiving
    if archiving_dir:
        archiving_dir = Path(archiving_dir)
        archiving_dir.mkdir(parents=True, exist_ok=True)
        submission_df.write_csv(archiving_dir / f'{wandb.run.name}_{output_filename}')

    return submission_df

# %%
if inference_config['create_submission_csv']:
    submission_df = create_submission_csv(ys_pred['thp_vol'])

# %%
# debug_submission_df_5w_6w = pl.DataFrame(
#     {'idx_hour': pl.Series(range(1, 1901))} | {id: pl.Series(range(1, 1901)) for id in ys_pred['thp_vol'].columns})
# debug_filtered = debug_submission_df_5w_6w.filter(pl.col('idx_hour') >= 841, pl.col('idx_hour') <= 1848)
# df = create_half_submission_df(debug_filtered, '5w-6w')
# # df = create_submission_csv(ys_pred['thp_vol'], 'traffic_forecast.csv', 'submission-csvs-archive')
# create_submission_csv(debug_filtered)

# %%


# %%



