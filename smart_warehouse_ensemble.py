import argparse
import gc
import json
import warnings
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

try:
    import lightgbm as lgb
    LIGHTGBM_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    lgb = None
    LIGHTGBM_IMPORT_ERROR = exc

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


SEED = 42
TARGET_COL = "avg_delay_minutes_next_30m"
ID_COL = "ID"
GROUP_COL = "scenario_id"
LAYOUT_COL = "layout_id"
TIME_COL = "time_idx"


def seed_everything(seed: int = SEED) -> None:
    np.random.seed(seed)


def require_lightgbm() -> None:
    if lgb is None:
        raise RuntimeError(
            "LightGBM could not be imported. On macOS this usually means libomp is missing. "
            "Install libomp and rerun. Original import error: "
            f"{LIGHTGBM_IMPORT_ERROR}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="/Users/ijaehyeong/Downloads/open/train.csv")
    parser.add_argument("--test-path", default="/Users/ijaehyeong/Downloads/open/test.csv")
    parser.add_argument("--layout-path", default="/Users/ijaehyeong/Downloads/open/layout_info.csv")
    parser.add_argument("--sample-path", default="/Users/ijaehyeong/Downloads/open/sample_submission.csv")
    parser.add_argument("--submission-path", default="submission.csv")
    parser.add_argument("--feature-importance-path", default="feature_importance.csv")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--selector-top-k", type=int, default=260)
    parser.add_argument("--selector-folds", type=int, default=3)
    parser.add_argument("--lgb-trials", type=int, default=18)
    parser.add_argument("--cat-trials", type=int, default=12)
    parser.add_argument("--stable-top-k", type=int, default=140)
    parser.add_argument("--disable-tuning", action="store_true")
    parser.add_argument("--verbosity", type=int, default=100)
    return parser.parse_args()


@dataclass
class DatasetBundle:
    train: pd.DataFrame
    test: pd.DataFrame
    feature_cols: List[str]
    categorical_cols: List[str]


def safe_divide(a: pd.Series, b: pd.Series, eps: float = 1e-6) -> pd.Series:
    return a / (b.abs() + eps)


def add_manual_features(df: pd.DataFrame) -> pd.DataFrame:
    df[TIME_COL] = df.groupby(GROUP_COL).cumcount().astype(np.int16)
    df["time_to_end"] = (24 - df[TIME_COL]).astype(np.int16)
    df["time_progress"] = (df[TIME_COL] / 24.0).astype(np.float32)
    df["time_bucket_5"] = (df[TIME_COL] // 5).astype(np.int16)
    df["is_early_phase"] = (df[TIME_COL] <= 4).astype(np.int8)
    df["is_peak_phase"] = df[TIME_COL].between(10, 19).astype(np.int8)
    df["is_late_phase"] = (df[TIME_COL] >= 20).astype(np.int8)

    if "shift_hour" in df.columns:
        hour = df["shift_hour"].fillna(df["shift_hour"].median())
        df["shift_hour_sin"] = np.sin(2 * np.pi * hour / 24.0).astype(np.float32)
        df["shift_hour_cos"] = np.cos(2 * np.pi * hour / 24.0).astype(np.float32)

    if "day_of_week" in df.columns:
        day = pd.to_numeric(df["day_of_week"], errors="coerce").fillna(0)
        df["day_of_week_sin"] = np.sin(2 * np.pi * day / 7.0).astype(np.float32)
        df["day_of_week_cos"] = np.cos(2 * np.pi * day / 7.0).astype(np.float32)

    manual_recipes = {
        "orders_per_robot_active": ("order_inflow_15m", "robot_active"),
        "orders_per_staff": ("order_inflow_15m", "staff_on_floor"),
        "orders_per_dock_util": ("order_inflow_15m", "loading_dock_util"),
        "orders_per_wave": ("order_inflow_15m", "order_wave_count"),
        "orders_per_unique_sku": ("order_inflow_15m", "unique_sku_15m"),
        "robot_load_gap": ("robot_active", "robot_idle"),
        "battery_margin": ("battery_mean", "battery_std"),
        "battery_load_pressure": ("battery_mean", "order_inflow_15m"),
        "low_battery_congestion": ("low_battery_ratio", "congestion_score"),
        "charge_pressure_ratio": ("charge_queue_length", "robot_charging"),
        "queue_per_robot": ("charge_queue_length", "robot_active"),
        "congestion_x_orders": ("congestion_score", "order_inflow_15m"),
        "congestion_x_reassign": ("congestion_score", "task_reassign_15m"),
        "congestion_x_robot_util": ("congestion_score", "robot_utilization"),
        "collision_fault_pressure": ("near_collision_15m", "fault_count_15m"),
        "pack_flow_pressure": ("pack_utilization", "order_inflow_15m"),
        "dock_flow_pressure": ("loading_dock_util", "outbound_truck_wait_min"),
        "backorder_wave_pressure": ("backorder_ratio", "order_wave_count"),
        "traffic_wait_pressure": ("aisle_traffic_score", "intersection_wait_time_avg"),
        "storage_density_x_flow": ("storage_density_pct", "order_inflow_15m"),
        "layout_robot_density": ("robot_total", "floor_area_sqm"),
        "layout_station_density": ("pack_station_count", "floor_area_sqm"),
    }

    for feat_name, (a_col, b_col) in manual_recipes.items():
        if a_col in df.columns and b_col in df.columns:
            a = pd.to_numeric(df[a_col], errors="coerce")
            b = pd.to_numeric(df[b_col], errors="coerce")
            if feat_name in {"robot_load_gap", "battery_margin"}:
                df[feat_name] = (a - b).astype(np.float32)
            elif feat_name in {
                "low_battery_congestion",
                "congestion_x_orders",
                "congestion_x_reassign",
                "congestion_x_robot_util",
                "collision_fault_pressure",
                "pack_flow_pressure",
                "dock_flow_pressure",
                "backorder_wave_pressure",
                "traffic_wait_pressure",
                "storage_density_x_flow",
            }:
                df[feat_name] = (a * b).astype(np.float32)
            else:
                df[feat_name] = safe_divide(a, b).astype(np.float32)

    if {"order_inflow_15m", "robot_total"}.issubset(df.columns):
        df["orders_per_total_robot"] = safe_divide(df["order_inflow_15m"], df["robot_total"]).astype(np.float32)
    if {"task_reassign_15m", "robot_active"}.issubset(df.columns):
        df["reassign_per_robot"] = safe_divide(df["task_reassign_15m"], df["robot_active"]).astype(np.float32)
    if {"battery_mean", "low_battery_ratio"}.issubset(df.columns):
        df["battery_reserve_index"] = (df["battery_mean"] * (1.0 - df["low_battery_ratio"])).astype(np.float32)
    if {"order_inflow_15m", "pack_station_count"}.issubset(df.columns):
        df["orders_per_pack_station"] = safe_divide(df["order_inflow_15m"], df["pack_station_count"]).astype(np.float32)
    if {"congestion_score", "robot_total"}.issubset(df.columns):
        df["congestion_per_robot_total"] = safe_divide(df["congestion_score"], df["robot_total"]).astype(np.float32)
    if {"charge_queue_length", "charger_count"}.issubset(df.columns):
        df["queue_per_charger"] = safe_divide(df["charge_queue_length"], df["charger_count"]).astype(np.float32)
    if {"staff_on_floor", "floor_area_sqm"}.issubset(df.columns):
        df["staff_density_layout"] = safe_divide(df["staff_on_floor"], df["floor_area_sqm"]).astype(np.float32)

    return df


def add_group_time_features(df: pd.DataFrame, base_cols: Sequence[str]) -> pd.DataFrame:
    lag_steps = [1, 2, 3, 5]
    windows = [3, 5, 7]
    group_obj = df.groupby(GROUP_COL, sort=False)

    for col in base_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        grouped = group_obj[col]

        for lag in lag_steps:
            df[f"{col}_lag_{lag}"] = grouped.shift(lag).astype(np.float32)

        shifted = grouped.shift(1)
        for window in windows:
            roll = shifted.groupby(df[GROUP_COL], sort=False).rolling(window=window, min_periods=1)
            df[f"{col}_roll_mean_{window}"] = roll.mean().reset_index(level=0, drop=True).astype(np.float32)
            df[f"{col}_roll_std_{window}"] = roll.std().reset_index(level=0, drop=True).astype(np.float32)

        df[f"{col}_diff_1"] = (series - grouped.shift(1)).astype(np.float32)
        df[f"{col}_diff_3"] = (series - grouped.shift(3)).astype(np.float32)
        df[f"{col}_trend_3"] = safe_divide(series - grouped.shift(3), pd.Series(3.0, index=df.index)).astype(np.float32)
        df[f"{col}_trend_5"] = safe_divide(series - grouped.shift(5), pd.Series(5.0, index=df.index)).astype(np.float32)
        df[f"{col}_recent_gap"] = (
            df[f"{col}_roll_mean_3"] - df[f"{col}_roll_mean_7"]
        ).astype(np.float32)
        df[f"{col}_level_vs_past"] = (series - df[f"{col}_roll_mean_5"]).astype(np.float32)

        expanding_shifted = shifted.groupby(df[GROUP_COL], sort=False).expanding(min_periods=1)
        df[f"{col}_exp_mean"] = expanding_shifted.mean().reset_index(level=0, drop=True).astype(np.float32)
        df[f"{col}_exp_std"] = expanding_shifted.std().reset_index(level=0, drop=True).astype(np.float32)
        first_value = grouped.transform("first")
        df[f"{col}_vs_start"] = (series - first_value).astype(np.float32)
        df[f"{col}_ratio_to_start"] = safe_divide(series, first_value).astype(np.float32)

    return df


def add_extended_time_features(df: pd.DataFrame, focus_cols: Sequence[str]) -> pd.DataFrame:
    long_lags = [8, 10, 12]
    group_obj = df.groupby(GROUP_COL, sort=False)

    for col in focus_cols:
        grouped = group_obj[col]
        series = pd.to_numeric(df[col], errors="coerce")
        shifted = grouped.shift(1)

        for lag in long_lags:
            df[f"{col}_lag_{lag}"] = grouped.shift(lag).astype(np.float32)

        df[f"{col}_recent2_mean"] = (
            shifted.groupby(df[GROUP_COL], sort=False).rolling(window=2, min_periods=1).mean().reset_index(level=0, drop=True)
        ).astype(np.float32)
        df[f"{col}_past2_mean"] = (
            grouped.shift(3).groupby(df[GROUP_COL], sort=False).rolling(window=2, min_periods=1).mean().reset_index(level=0, drop=True)
        ).astype(np.float32)
        df[f"{col}_recent_past2_gap"] = (df[f"{col}_recent2_mean"] - df[f"{col}_past2_mean"]).astype(np.float32)
        df[f"{col}_roll_max_5"] = (
            shifted.groupby(df[GROUP_COL], sort=False).rolling(window=5, min_periods=1).max().reset_index(level=0, drop=True)
        ).astype(np.float32)
        df[f"{col}_roll_min_5"] = (
            shifted.groupby(df[GROUP_COL], sort=False).rolling(window=5, min_periods=1).min().reset_index(level=0, drop=True)
        ).astype(np.float32)
        df[f"{col}_roll_range_5"] = (df[f"{col}_roll_max_5"] - df[f"{col}_roll_min_5"]).astype(np.float32)
        df[f"{col}_ewm_mean_3"] = shifted.groupby(df[GROUP_COL], sort=False).transform(
            lambda x: x.ewm(span=3, adjust=False, min_periods=1).mean()
        ).astype(np.float32)
        df[f"{col}_ewm_mean_5"] = shifted.groupby(df[GROUP_COL], sort=False).transform(
            lambda x: x.ewm(span=5, adjust=False, min_periods=1).mean()
        ).astype(np.float32)
        df[f"{col}_ewm_gap"] = (df[f"{col}_ewm_mean_3"] - df[f"{col}_ewm_mean_5"]).astype(np.float32)
        df[f"{col}_cur_to_longlag12"] = (series - grouped.shift(12)).astype(np.float32)

    return df


def add_layout_time_aggregates(df: pd.DataFrame, agg_cols: Sequence[str]) -> pd.DataFrame:
    for group_keys, suffix in [
        ([LAYOUT_COL, TIME_COL], "layout_time"),
        (["layout_type", TIME_COL], "layouttype_time"),
        ([LAYOUT_COL, "time_bucket_5"], "layout_bucket"),
    ]:
        if not set(group_keys).issubset(df.columns):
            continue
        for col in agg_cols:
            group_series = df.groupby(group_keys, sort=False)[col]
            mean_col = f"{col}_{suffix}_mean"
            std_col = f"{col}_{suffix}_std"
            df[mean_col] = group_series.transform("mean").astype(np.float32)
            df[std_col] = group_series.transform("std").astype(np.float32)
            df[f"{col}_{suffix}_delta"] = (pd.to_numeric(df[col], errors="coerce") - df[mean_col]).astype(np.float32)
            df[f"{col}_{suffix}_z"] = safe_divide(df[f"{col}_{suffix}_delta"], df[std_col].fillna(0)).astype(np.float32)
    return df


def reduce_memory(df: pd.DataFrame, categorical_cols: Sequence[str]) -> pd.DataFrame:
    for col in df.columns:
        if col in categorical_cols:
            continue
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype(np.float32)
        elif pd.api.types.is_integer_dtype(df[col]):
            if df[col].min() >= np.iinfo(np.int16).min and df[col].max() <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            else:
                df[col] = df[col].astype(np.int32)
    return df


def build_features(train: pd.DataFrame, test: pd.DataFrame, layout: pd.DataFrame) -> DatasetBundle:
    train = train.copy()
    test = test.copy()
    train["_is_train"] = 1
    test["_is_train"] = 0
    test[TARGET_COL] = np.nan

    combined = pd.concat([train, test], axis=0, ignore_index=True)
    combined = combined.merge(layout, on=LAYOUT_COL, how="left")
    combined = add_manual_features(combined)

    key_dynamic_cols = [
        "order_inflow_15m",
        "unique_sku_15m",
        "avg_items_per_order",
        "urgent_order_ratio",
        "robot_active",
        "robot_utilization",
        "task_reassign_15m",
        "battery_mean",
        "low_battery_ratio",
        "charge_queue_length",
        "avg_charge_wait",
        "congestion_score",
        "max_zone_density",
        "near_collision_15m",
        "fault_count_15m",
        "pack_utilization",
        "staff_on_floor",
        "loading_dock_util",
        "order_wave_count",
        "pick_list_length_avg",
        "aisle_traffic_score",
        "intersection_wait_time_avg",
        "outbound_truck_wait_min",
        "dock_to_stock_hours",
        "backorder_ratio",
        "shift_handover_delay_min",
        "sort_accuracy_pct",
    ]
    key_dynamic_cols = [c for c in key_dynamic_cols if c in combined.columns]
    combined = add_group_time_features(combined, key_dynamic_cols)
    extended_focus_cols = [
        "order_inflow_15m",
        "robot_utilization",
        "congestion_score",
        "battery_mean",
        "charge_queue_length",
        "task_reassign_15m",
        "pack_utilization",
        "outbound_truck_wait_min",
    ]
    extended_focus_cols = [c for c in extended_focus_cols if c in combined.columns]
    combined = add_extended_time_features(combined, extended_focus_cols)
    layout_agg_cols = [
        "order_inflow_15m",
        "robot_utilization",
        "congestion_score",
        "battery_mean",
        "charge_queue_length",
        "pack_utilization",
        "outbound_truck_wait_min",
        "staff_on_floor",
        "loading_dock_util",
    ]
    layout_agg_cols = [c for c in layout_agg_cols if c in combined.columns]
    combined = add_layout_time_aggregates(combined, layout_agg_cols)

    categorical_cols = [col for col in [LAYOUT_COL, "layout_type", "day_of_week", "time_bucket_5"] if col in combined.columns]
    for col in categorical_cols:
        combined[col] = combined[col].fillna("missing").astype(str).astype("category")

    protected_cols = {ID_COL, GROUP_COL, TARGET_COL, "_is_train"}
    feature_cols = [col for col in combined.columns if col not in protected_cols]
    combined = reduce_memory(combined, categorical_cols)

    train_feat = combined[combined["_is_train"] == 1].drop(columns=["_is_train"]).reset_index(drop=True)
    test_feat = combined[combined["_is_train"] == 0].drop(columns=["_is_train", TARGET_COL]).reset_index(drop=True)

    return DatasetBundle(
        train=train_feat,
        test=test_feat,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
    )


def prepare_target(y: pd.Series) -> np.ndarray:
    return np.log1p(y.values.astype(np.float32))


def inverse_target(y: np.ndarray) -> np.ndarray:
    return np.clip(np.expm1(y), 0, None)


def get_folds(groups: pd.Series, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    splitter = GroupKFold(n_splits=n_splits)
    dummy = np.zeros(len(groups), dtype=np.float32)
    return list(splitter.split(dummy, groups=groups, y=dummy))


def fit_selector(
    X: pd.DataFrame,
    y_log: np.ndarray,
    groups: pd.Series,
    feature_cols: Sequence[str],
    categorical_cols: Sequence[str],
    selector_folds: int,
    top_k: int,
) -> Tuple[List[str], pd.DataFrame]:
    require_lightgbm()
    folds = get_folds(groups, selector_folds)
    importance_acc = pd.Series(0.0, index=list(feature_cols))

    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.03,
        "num_leaves": 96,
        "max_depth": -1,
        "min_data_in_leaf": 40,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.3,
        "lambda_l2": 2.0,
        "seed": SEED,
        "verbosity": -1,
        "n_estimators": 1500,
        "n_jobs": 1,
    }

    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        X_tr = X.iloc[tr_idx][list(feature_cols)].copy()
        X_va = X.iloc[va_idx][list(feature_cols)].copy()
        y_tr, y_va = y_log[tr_idx], y_log[va_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="l1",
            categorical_feature=[c for c in categorical_cols if c in feature_cols],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        importance_acc += pd.Series(model.booster_.feature_importance(importance_type="gain"), index=list(feature_cols))

    importance_df = (
        importance_acc.sort_values(ascending=False)
        .rename("gain_importance")
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    selected = importance_df["feature"].head(top_k).tolist()
    for col in categorical_cols:
        if col in feature_cols and col not in selected:
            selected.append(col)

    return selected, importance_df


def score_mae(y_true: np.ndarray, y_pred_log: np.ndarray) -> float:
    return mean_absolute_error(y_true, inverse_target(y_pred_log))


def score_mae_raw(y_true: np.ndarray, y_pred_raw: np.ndarray) -> float:
    return mean_absolute_error(y_true, np.clip(y_pred_raw, 0, None))


def tune_lightgbm(
    X: pd.DataFrame,
    y: np.ndarray,
    y_log: np.ndarray,
    groups: pd.Series,
    feature_cols: Sequence[str],
    categorical_cols: Sequence[str],
    n_trials: int,
    verbosity: int,
) -> Dict[str, float]:
    require_lightgbm()
    if n_trials <= 0:
        return {}

    folds = get_folds(groups, min(3, groups.nunique()))
    folds = folds[:3]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.06, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 48, 180),
            "max_depth": trial.suggest_int("max_depth", 5, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.65, 0.98),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.65, 0.98),
            "bagging_freq": 1,
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 20.0, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.2),
            "max_bin": trial.suggest_int("max_bin", 255, 511),
            "seed": SEED,
            "verbosity": -1,
            "n_estimators": 4000,
            "n_jobs": 1,
        }

        fold_scores = []
        for tr_idx, va_idx in folds:
            X_tr = X.iloc[tr_idx][list(feature_cols)].copy()
            X_va = X.iloc[va_idx][list(feature_cols)].copy()
            y_tr, y_va = y_log[tr_idx], y_log[va_idx]

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="l1",
                categorical_feature=[c for c in categorical_cols if c in feature_cols],
                callbacks=[lgb.early_stopping(150, verbose=False)],
            )
            pred = model.predict(X_va, num_iteration=model.best_iteration_)
            fold_scores.append(score_mae(y[va_idx], pred))
        return float(np.mean(fold_scores))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbosity > 0)
    return study.best_params


def tune_catboost(
    X: pd.DataFrame,
    y: np.ndarray,
    y_log: np.ndarray,
    groups: pd.Series,
    feature_cols: Sequence[str],
    categorical_cols: Sequence[str],
    n_trials: int,
    verbosity: int,
) -> Dict[str, float]:
    if n_trials <= 0:
        return {}

    folds = get_folds(groups, min(3, groups.nunique()))
    folds = folds[:3]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "loss_function": "MAE",
            "eval_metric": "MAE",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
            "depth": trial.suggest_int("depth", 5, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 12.0),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 5.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 4.0),
            "subsample": trial.suggest_float("subsample", 0.65, 0.98),
            "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise"]),
            "iterations": 4000,
            "random_seed": SEED,
            "verbose": False,
            "allow_writing_files": False,
        }

        fold_scores = []
        for tr_idx, va_idx in folds:
            X_tr = X.iloc[tr_idx][list(feature_cols)].copy()
            X_va = X.iloc[va_idx][list(feature_cols)].copy()
            y_tr, y_va = y_log[tr_idx], y_log[va_idx]

            tr_pool = Pool(X_tr, y_tr, cat_features=[c for c in categorical_cols if c in feature_cols])
            va_pool = Pool(X_va, y_va, cat_features=[c for c in categorical_cols if c in feature_cols])
            model = CatBoostRegressor(**params)
            model.fit(tr_pool, eval_set=va_pool, use_best_model=True, early_stopping_rounds=150)
            pred = model.predict(X_va)
            fold_scores.append(score_mae(y[va_idx], pred))
        return float(np.mean(fold_scores))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbosity > 0)
    return study.best_params


def train_lightgbm_cv(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y: np.ndarray,
    y_log: np.ndarray,
    groups: pd.Series,
    feature_cols: Sequence[str],
    categorical_cols: Sequence[str],
    params_override: Dict[str, float],
    n_splits: int,
) -> Tuple[np.ndarray, np.ndarray, List[float], pd.Series]:
    require_lightgbm()
    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.02,
        "num_leaves": 96,
        "max_depth": 8,
        "min_data_in_leaf": 48,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.84,
        "bagging_freq": 1,
        "lambda_l1": 0.15,
        "lambda_l2": 3.5,
        "min_gain_to_split": 0.0,
        "max_bin": 255,
        "seed": SEED,
        "verbosity": -1,
        "n_estimators": 6000,
        "n_jobs": 1,
    }
    params.update(params_override)

    folds = get_folds(groups, n_splits)
    oof = np.zeros(len(X_train), dtype=np.float32)
    test_pred = np.zeros(len(X_test), dtype=np.float32)
    fold_scores = []
    importance_acc = pd.Series(0.0, index=list(feature_cols))

    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        X_tr = X_train.iloc[tr_idx][list(feature_cols)].copy()
        X_va = X_train.iloc[va_idx][list(feature_cols)].copy()
        y_tr, y_va = y_log[tr_idx], y_log[va_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="l1",
            categorical_feature=[c for c in categorical_cols if c in feature_cols],
            callbacks=[lgb.early_stopping(250, verbose=False)],
        )

        oof[va_idx] = model.predict(X_va, num_iteration=model.best_iteration_)
        test_pred += model.predict(X_test[list(feature_cols)], num_iteration=model.best_iteration_) / n_splits
        fold_mae = score_mae(y[va_idx], oof[va_idx])
        fold_scores.append(fold_mae)
        importance_acc += pd.Series(model.booster_.feature_importance(importance_type="gain"), index=list(feature_cols))
        print(f"[LightGBM] Fold {fold}: MAE={fold_mae:.5f}, best_iter={model.best_iteration_}")

    return oof, test_pred, fold_scores, importance_acc


def train_catboost_cv(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y: np.ndarray,
    y_log: np.ndarray,
    groups: pd.Series,
    feature_cols: Sequence[str],
    categorical_cols: Sequence[str],
    params_override: Dict[str, float],
    n_splits: int,
) -> Tuple[np.ndarray, np.ndarray, List[float], pd.Series]:
    params = {
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "learning_rate": 0.02,
        "depth": 8,
        "l2_leaf_reg": 5.0,
        "random_strength": 0.5,
        "bagging_temperature": 1.0,
        "subsample": 0.85,
        "grow_policy": "SymmetricTree",
        "iterations": 6000,
        "random_seed": SEED,
        "verbose": False,
        "allow_writing_files": False,
    }
    params.update(params_override)

    folds = get_folds(groups, n_splits)
    oof = np.zeros(len(X_train), dtype=np.float32)
    test_pred = np.zeros(len(X_test), dtype=np.float32)
    fold_scores = []
    importance_acc = pd.Series(0.0, index=list(feature_cols))

    cat_features = [c for c in categorical_cols if c in feature_cols]

    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        X_tr = X_train.iloc[tr_idx][list(feature_cols)].copy()
        X_va = X_train.iloc[va_idx][list(feature_cols)].copy()
        y_tr, y_va = y_log[tr_idx], y_log[va_idx]

        tr_pool = Pool(X_tr, y_tr, cat_features=cat_features)
        va_pool = Pool(X_va, y_va, cat_features=cat_features)

        model = CatBoostRegressor(**params)
        model.fit(tr_pool, eval_set=va_pool, use_best_model=True, early_stopping_rounds=250)

        oof[va_idx] = model.predict(X_va)
        test_pred += model.predict(X_test[list(feature_cols)]) / n_splits
        fold_mae = score_mae(y[va_idx], oof[va_idx])
        fold_scores.append(fold_mae)
        importance_acc += pd.Series(model.get_feature_importance(tr_pool), index=list(feature_cols))
        print(f"[CatBoost] Fold {fold}: MAE={fold_mae:.5f}, best_iter={model.get_best_iteration()}")

    return oof, test_pred, fold_scores, importance_acc


def train_catboost_cv_raw(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series,
    feature_cols: Sequence[str],
    categorical_cols: Sequence[str],
    params_override: Dict[str, float],
    n_splits: int,
) -> Tuple[np.ndarray, np.ndarray, List[float], pd.Series]:
    params = {
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "learning_rate": 0.03,
        "depth": 7,
        "l2_leaf_reg": 4.0,
        "random_strength": 0.35,
        "bagging_temperature": 0.5,
        "subsample": 0.9,
        "grow_policy": "SymmetricTree",
        "iterations": 5000,
        "random_seed": SEED,
        "verbose": False,
        "allow_writing_files": False,
    }
    params.update(params_override)

    folds = get_folds(groups, n_splits)
    oof = np.zeros(len(X_train), dtype=np.float32)
    test_pred = np.zeros(len(X_test), dtype=np.float32)
    fold_scores = []
    importance_acc = pd.Series(0.0, index=list(feature_cols))
    cat_features = [c for c in categorical_cols if c in feature_cols]

    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        X_tr = X_train.iloc[tr_idx][list(feature_cols)].copy()
        X_va = X_train.iloc[va_idx][list(feature_cols)].copy()
        y_tr, y_va = y[tr_idx], y[va_idx]

        tr_pool = Pool(X_tr, y_tr, cat_features=cat_features)
        va_pool = Pool(X_va, y_va, cat_features=cat_features)

        model = CatBoostRegressor(**params)
        model.fit(tr_pool, eval_set=va_pool, use_best_model=True, early_stopping_rounds=200)

        oof[va_idx] = np.clip(model.predict(X_va), 0, None)
        test_pred += np.clip(model.predict(X_test[list(feature_cols)]), 0, None) / n_splits
        fold_mae = score_mae_raw(y[va_idx], oof[va_idx])
        fold_scores.append(fold_mae)
        importance_acc += pd.Series(model.get_feature_importance(tr_pool), index=list(feature_cols))
        print(f"[CatBoostRaw] Fold {fold}: MAE={fold_mae:.5f}, best_iter={model.get_best_iteration()}")

    return oof, test_pred, fold_scores, importance_acc


def build_feature_subsets(
    selected_features: Sequence[str],
    selector_importance: pd.DataFrame,
    categorical_cols: Sequence[str],
    stable_top_k: int,
) -> Dict[str, List[str]]:
    selected_set = set(selected_features)
    stable = selector_importance.loc[selector_importance["feature"].isin(selected_set), "feature"].head(stable_top_k).tolist()
    for col in categorical_cols:
        if col in selected_set and col not in stable:
            stable.append(col)

    time_focus_keywords = ("lag_", "roll_", "ewm_", "trend_", "recent_", "time_", "layout_time", "layout_bucket")
    time_focus = [
        col for col in selected_features
        if any(key in col for key in time_focus_keywords) or col in categorical_cols
    ]
    if len(time_focus) < 120:
        time_focus = list(dict.fromkeys(time_focus + list(selected_features)[:120]))

    return {
        "primary": list(selected_features),
        "stable": stable,
        "time_focus": time_focus,
    }


def optimize_ensemble_weights_raw(
    y_true: np.ndarray,
    pred_dict: Dict[str, np.ndarray],
) -> Tuple[Dict[str, float], float]:
    names = list(pred_dict.keys())
    best_weights = {name: 1.0 / len(names) for name in names}
    best_score = score_mae_raw(y_true, np.mean(np.column_stack([pred_dict[name] for name in names]), axis=1))

    if len(names) == 2:
        for w in np.linspace(0.05, 0.95, 37):
            weights = {names[0]: float(w), names[1]: float(1.0 - w)}
            blend = sum(weights[name] * pred_dict[name] for name in names)
            score = score_mae_raw(y_true, blend)
            if score < best_score:
                best_score = score
                best_weights = weights
        return best_weights, best_score

    if len(names) == 3:
        grid = np.linspace(0.05, 0.90, 18)
        for w1 in grid:
            for w2 in grid:
                if w1 + w2 >= 0.95:
                    continue
                w3 = 1.0 - w1 - w2
                if w3 <= 0.05:
                    continue
                weights = {names[0]: float(w1), names[1]: float(w2), names[2]: float(w3)}
                blend = sum(weights[name] * pred_dict[name] for name in names)
                score = score_mae_raw(y_true, blend)
                if score < best_score:
                    best_score = score
                    best_weights = weights
        return best_weights, best_score

    return best_weights, best_score


def main() -> None:
    args = parse_args()
    seed_everything(SEED)

    train = pd.read_csv(args.train_path)
    test = pd.read_csv(args.test_path)
    layout = pd.read_csv(args.layout_path)
    sample_submission = pd.read_csv(args.sample_path)

    print("Building leakage-safe time-series features...", flush=True)
    data = build_features(train, test, layout)
    print("Feature engineering complete.", flush=True)
    train_feat = data.train
    test_feat = data.test

    y = train_feat[TARGET_COL].values.astype(np.float32)
    y_log = prepare_target(train_feat[TARGET_COL])
    groups = train_feat[GROUP_COL]

    feature_cols = [c for c in data.feature_cols if c in train_feat.columns and c in test_feat.columns]
    categorical_cols = [c for c in data.categorical_cols if c in feature_cols]

    print(f"Train shape after feature engineering: {train_feat[feature_cols].shape}")
    print(f"Test shape after feature engineering:  {test_feat[feature_cols].shape}")
    print(f"Categorical features: {categorical_cols}")

    selected_features, selector_importance = fit_selector(
        X=train_feat,
        y_log=y_log,
        groups=groups,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        selector_folds=args.selector_folds,
        top_k=args.selector_top_k,
    )
    print(f"Selected {len(selected_features)} features after importance pruning")

    lgb_best_params = {}
    cat_best_params = {}
    if not args.disable_tuning:
        print("Running Optuna for LightGBM...")
        lgb_best_params = tune_lightgbm(
            X=train_feat,
            y=y,
            y_log=y_log,
            groups=groups,
            feature_cols=selected_features,
            categorical_cols=categorical_cols,
            n_trials=args.lgb_trials,
            verbosity=args.verbosity,
        )
        print("Best LightGBM params:", json.dumps(lgb_best_params, indent=2))

        print("Running Optuna for CatBoost...")
        cat_best_params = tune_catboost(
            X=train_feat,
            y=y,
            y_log=y_log,
            groups=groups,
            feature_cols=selected_features,
            categorical_cols=categorical_cols,
            n_trials=args.cat_trials,
            verbosity=args.verbosity,
        )
        print("Best CatBoost params:", json.dumps(cat_best_params, indent=2))

    feature_sets = build_feature_subsets(
        selected_features=selected_features,
        selector_importance=selector_importance,
        categorical_cols=categorical_cols,
        stable_top_k=args.stable_top_k,
    )
    print(
        "Feature subsets sizes:",
        {name: len(cols) for name, cols in feature_sets.items()},
    )

    lgb_oof, lgb_test, lgb_scores, lgb_imp = train_lightgbm_cv(
        X_train=train_feat,
        X_test=test_feat,
        y=y,
        y_log=y_log,
        groups=groups,
        feature_cols=feature_sets["primary"],
        categorical_cols=categorical_cols,
        params_override=lgb_best_params,
        n_splits=args.n_splits,
    )
    lgb_cv = mean_absolute_error(y, inverse_target(lgb_oof))
    print(f"LightGBM CV MAE: {lgb_cv:.5f}")

    gc.collect()

    cat_oof, cat_test, cat_scores, cat_imp = train_catboost_cv(
        X_train=train_feat,
        X_test=test_feat,
        y=y,
        y_log=y_log,
        groups=groups,
        feature_cols=feature_sets["primary"],
        categorical_cols=categorical_cols,
        params_override=cat_best_params,
        n_splits=args.n_splits,
    )
    cat_cv = mean_absolute_error(y, inverse_target(cat_oof))
    print(f"CatBoost CV MAE: {cat_cv:.5f}")

    gc.collect()

    cat_raw_oof, cat_raw_test, cat_raw_scores, cat_raw_imp = train_catboost_cv_raw(
        X_train=train_feat,
        X_test=test_feat,
        y=y,
        groups=groups,
        feature_cols=feature_sets["stable"],
        categorical_cols=categorical_cols,
        params_override={},
        n_splits=args.n_splits,
    )
    cat_raw_cv = score_mae_raw(y, cat_raw_oof)
    print(f"CatBoost Raw CV MAE: {cat_raw_cv:.5f}")

    raw_pred_dict = {
        "lgb": inverse_target(lgb_oof),
        "cat": inverse_target(cat_oof),
        "cat_raw": cat_raw_oof,
    }
    raw_test_pred_dict = {
        "lgb": inverse_target(lgb_test),
        "cat": inverse_target(cat_test),
        "cat_raw": cat_raw_test,
    }
    ensemble_weights, ensemble_cv = optimize_ensemble_weights_raw(y, raw_pred_dict)
    print(f"Ensemble weights -> {ensemble_weights}")
    print(f"Ensemble CV MAE: {ensemble_cv:.5f}")

    final_test_pred = np.zeros(len(test_feat), dtype=np.float32)
    for name, weight in ensemble_weights.items():
        final_test_pred += weight * raw_test_pred_dict[name]
    sample_submission[TARGET_COL] = final_test_pred
    sample_submission.to_csv(args.submission_path, index=False)
    print(f"Saved submission to {args.submission_path}")

    feature_importance = (
        selector_importance.set_index("feature")
        .join(lgb_imp.rename("lgb_gain"), how="outer")
        .join(cat_imp.rename("cat_importance"), how="outer")
        .join(cat_raw_imp.rename("cat_raw_importance"), how="outer")
        .fillna(0.0)
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    feature_importance["combined_rank_score"] = (
        feature_importance["gain_importance"].rank(ascending=False, method="average")
        + feature_importance["lgb_gain"].rank(ascending=False, method="average")
        + feature_importance["cat_importance"].rank(ascending=False, method="average")
        + feature_importance["cat_raw_importance"].rank(ascending=False, method="average")
    )
    feature_importance = feature_importance.sort_values("combined_rank_score")
    feature_importance.to_csv(args.feature_importance_path, index=False)
    print(f"Saved feature importance to {args.feature_importance_path}")

    print("Fold MAE summary")
    print("LightGBM:", [round(v, 5) for v in lgb_scores])
    print("CatBoost:", [round(v, 5) for v in cat_scores])
    print("CatBoostRaw:", [round(v, 5) for v in cat_raw_scores])


if __name__ == "__main__":
    main()
