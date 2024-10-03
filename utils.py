"""
Utility functions for traffic forecasting.
"""

import logging
import polars as pl
from pathlib import Path
from tqdm.notebook import tqdm


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

def make_base_dataframes(target_dataframes: dict[str, pl.DataFrame], base_df_names: list[str]) -> dict[str, pl.DataFrame]:
    """
    Create new dataframes to be used for feature engineering or as targets.
    """
    base_dataframes = {}
    for name in base_df_names:
        try:
            base_dataframes[name] = target_dataframes[name]
        except KeyError:
            base_dataframes[name] = make_interaction_dataframe(name, target_dataframes)
    return base_dataframes

def create_hour_feats(idx_hour_df: pl.DataFrame, lags: list[int]) -> dict[str, pl.DataFrame]:
    """
    Create shifted versions of daily 24h count for each column in the DataFrame.
    Returns a dictionary of DataFrames with the key format 'daily_hours_shifted_{lag}h'.
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

def create_time_feature_dfs(idx_hour_df: pl.DataFrame, idx_hour_shifts: list[int], weekday_shifts: list[int]) -> dict[str, pl.DataFrame]:
    """
    Create repeating 24h and 7d features.
    """
    hour_feats = create_hour_feats(idx_hour_df, idx_hour_shifts)
    weekday_feats = create_weekday_feats(idx_hour_df, weekday_shifts)

    return {**hour_feats, **weekday_feats}

def create_ts_feature_dfs(df_name: str, df: pl.DataFrame, lags: list[int], rolling_avgs: list[int], delta_reference_points: list[tuple[int, int]], std_windows: list[int], num_zeros_windows: list[int]) -> dict[str, pl.DataFrame]:
    """
    Create lag, rolling average, delta, and standard deviation features for all columns in the DataFrame.
    Returns a dict of DataFrames.
    """
    lag_feats = {f"{df_name}_lag_{lag}": df.shift(lag) for lag in lags}

    delta_feats = {f"{df_name}_delta_{point_pair[0]}_{point_pair[1]}": (df.shift(point_pair[0]) - df.shift(point_pair[1])) for point_pair in delta_reference_points}

    # We need to shift one step to avoid data leakage
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
        f"{df_name}_num_zeros_{window}":
            pl.DataFrame({
                col: (shifted_df[col] == 0).cast(pl.Float32).rolling_sum(window_size=window)
                for col in df.columns
            })
        for window in num_zeros_windows
    }

    return {**lag_feats, **rolling_avg_feats, **delta_feats, **std_feats, **num_zeros_feats}

def extract_cell_id(beam_id: str) -> str:
    """
    Extract the cell ID from a beam ID.
    """
    return "_".join(beam_id.split("_")[0:2])

def extract_station_id(beam_id: str) -> int:
    """
    Extract the station ID from a beam ID.
    """
    return int(beam_id.split("_")[0])

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
    station_cols = [col for col in dataframe.columns if col.startswith(str(station_id))]
    station_data = dataframe.select(station_cols)
    return station_data.mean(axis=1)

def create_id_feature_dfs(template_df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """
    Create DataFrames for beam, cell, and station IDs based on a template DataFrame.
    """
    id_feature_dfs = {}
    id_feature_dfs['beam_id'] = pl.DataFrame({beam_id: [beam_id] * len(template_df) for beam_id in template_df.columns})
    id_feature_dfs['cell_id'] = pl.DataFrame({beam_id: [extract_cell_id(beam_id)] * len(template_df) for beam_id in template_df.columns})
    id_feature_dfs['station_id'] = pl.DataFrame({beam_id: [extract_station_id(beam_id)] * len(template_df) for beam_id in template_df.columns})

    return id_feature_dfs

def create_all_feature_dfs(target_dataframes: dict[str, pl.DataFrame], idx_hour_series: pl.Series, config: dict) -> dict[str, pl.DataFrame]:
    """
    Create features for the traffic forecasting model for all beams at once.
    Returns a dictionary of feature DataFrames.
    """
    feature_dfs = {}
    template_df = target_dataframes['thp_vol']
    beam_ids = template_df.columns

    # Beam ID features
    # feature_dfs['beam_id'] = pl.DataFrame({beam_id: [int(beam_id)] * len(template_df) for beam_id in beam_ids})
    
    # Create a DataFrame full of the idx_hour series repeated across all columns
    idx_hour_df = pl.DataFrame({beam_id: idx_hour_series for beam_id in beam_ids})

    # Repeating 24h and 7d features
    feature_dfs.update(create_time_feature_dfs(
        idx_hour_df, config['hour_shifts'], config['weekday_shifts']))
    
    # Make list of dataframes, on which ts features will be created
    base_dataframes = make_base_dataframes(
        target_dataframes, config['feat_base_df_names'])
    
    for df_name, df in base_dataframes.items():
        logging.debug(f"Creating TS features for {df_name}")
        feature_dfs.update(create_ts_feature_dfs(df_name, df, config['lags'], config['rolling_avgs'],
                           config['delta_reference_points'], config['std_windows'], config['num_zeros_windows']))
    
    return feature_dfs

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
    Convert DataFrames to a dict of wide format DataFrames.
    Needs to have 'beam_id' and 'idx_hour' columns. 
    """
    wide_dfs = {}
    for df_name in output_df_names:
        wide_df = dataframe.pivot(index='idx_hour', columns='beam_id', values=df_name).drop('idx_hour')
        wide_dfs[df_name] = wide_df
    return wide_dfs

def create_long_format_df(target_dataframes: dict[str, pl.DataFrame], idx_hour_series: pl.Series, config: dict) -> pl.DataFrame:
    """
    Create long DataFrame with features and target columns.
    """
    feature_dfs = create_all_feature_dfs(target_dataframes, idx_hour_series, config)
    target_dfs = {k: v for k, v in target_dataframes.items()}
    all_train_dfs = {**feature_dfs, **target_dfs}
    long_train_df = convert_to_long_format(all_train_dfs)
    return long_train_df
