#!/usr/bin/env python
"""
Detecting DDoS Attacks using LSTM on the CIC-DDoS2019 dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from torch import nn
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
DEFAULT_CHUNK_SIZE = 50_000
MAX_CORR_FEATURES = 35
PLOT_DPI = 180


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "unnamed"


def standardize_columns(columns: Sequence[str]) -> List[str]:
    seen: Dict[str, int] = {}
    cleaned = []
    for column in columns:
        base = sanitize_name(str(column))
        count = seen.get(base, 0)
        new_name = f"{base}_{count}" if count else base
        seen[base] = count + 1
        cleaned.append(new_name)
    return cleaned


def infer_target_column(columns: Sequence[str]) -> str:
    priority = ["label", "target", "class", "attack", "category"]
    normalized = {sanitize_name(col): col for col in columns}
    for candidate in priority:
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError("Unable to infer target column automatically.")


def find_duplicate_columns(df: pd.DataFrame) -> List[str]:
    duplicate_columns: List[str] = []
    hashes: Dict[Tuple[str, ...], str] = {}
    for column in df.columns:
        signature = tuple(df[column].astype(str).fillna("<NA>").head(5000).tolist())
        if signature in hashes:
            duplicate_columns.append(column)
        else:
            hashes[signature] = column
    return duplicate_columns


def identify_leakage_columns(columns: Sequence[str]) -> Dict[str, str]:
    reasons: Dict[str, str] = {}
    keywords = {
        "flow_id": "Unique flow identifier is not deployable at inference time.",
        "source_ip": "IP addresses are environment-specific identifiers with high leakage risk.",
        "destination_ip": "IP addresses are environment-specific identifiers with high leakage risk.",
        "timestamp": "Timestamp is retained only for ordering, not as a predictive feature.",
        "simillarhttp": "HTTP similarity token is sparse and scenario-specific.",
        "unnamed_0": "Unnamed index-like column is not a real feature.",
        "source_port": "Port values can proxy attack family in this benchmark and may overstate generalization.",
        "destination_port": "Port values can proxy attack family in this benchmark and may overstate generalization.",
    }
    for column in columns:
        normalized = sanitize_name(column)
        if normalized in keywords:
            reasons[column] = keywords[normalized]
    return reasons


def list_csv_files(base_dir: Path) -> List[Path]:
    return sorted(base_dir.glob("*/*.csv"))


@dataclass
class PhaseConfig:
    name: str
    description: str
    relative_files: List[str]
    max_rows_total: int
    target_mode: str
    sequence_lengths: List[int]
    lstm_hidden_sizes: List[int]
    learning_rates: List[float]
    batch_sizes: List[int]
    dropout_values: List[float]
    oversample_training: bool = False
    split_strategy: str = "row_stratified"
    group_column: Optional[str] = None
    test_ratio: float = 0.15


def default_phase_configs() -> List[PhaseConfig]:
    return [
        PhaseConfig(
            name="phase1_dns_binary",
            description="Phase 1: Validate the full pipeline on a single DNS-focused subset.",
            relative_files=["01-12/DrDoS_DNS.csv"],
            max_rows_total=80_000,
            target_mode="binary",
            sequence_lengths=[5, 10],
            lstm_hidden_sizes=[32, 64],
            learning_rates=[1e-3],
            batch_sizes=[256],
            dropout_values=[0.2, 0.35],
            oversample_training=True,
        ),
        PhaseConfig(
            name="phase2_multiclass_subset",
            description="Phase 2: Expand to multiple attack families while keeping the run manageable.",
            relative_files=[
                "01-12/DrDoS_DNS.csv",
                "01-12/DrDoS_NTP.csv",
                "01-12/DrDoS_NetBIOS.csv",
                "03-11/Portmap.csv",
            ],
            max_rows_total=200_000,
            target_mode="multiclass",
            sequence_lengths=[5, 10],
            lstm_hidden_sizes=[64],
            learning_rates=[1e-3],
            batch_sizes=[512],
            dropout_values=[0.2, 0.35],
            oversample_training=False,
        ),
        PhaseConfig(
            name="phase3_scaling_subset",
            description="Phase 3: Scale the same pipeline to a larger, chunked subset across many files.",
            relative_files=[
                "01-12/DrDoS_DNS.csv",
                "01-12/DrDoS_LDAP.csv",
                "01-12/DrDoS_NTP.csv",
                "01-12/DrDoS_SSDP.csv",
                "01-12/Syn.csv",
                "03-11/Portmap.csv",
                "03-11/UDP.csv",
                "03-11/UDPLag.csv",
            ],
            max_rows_total=260_000,
            target_mode="multiclass",
            sequence_lengths=[10],
            lstm_hidden_sizes=[64],
            learning_rates=[5e-4],
            batch_sizes=[512],
            dropout_values=[0.35],
            oversample_training=False,
        ),
        PhaseConfig(
            name="phase25_multiclass_groupaware",
            description="Phase 2.5: Re-evaluate the multiclass setup with a strict source_ip-aware split.",
            relative_files=[
                "01-12/DrDoS_DNS.csv",
                "01-12/DrDoS_NTP.csv",
                "01-12/DrDoS_NetBIOS.csv",
                "03-11/Portmap.csv",
            ],
            max_rows_total=200_000,
            target_mode="multiclass",
            sequence_lengths=[5, 10],
            lstm_hidden_sizes=[64],
            learning_rates=[1e-3],
            batch_sizes=[512],
            dropout_values=[0.2, 0.35],
            oversample_training=False,
            split_strategy="group_aware",
            group_column="source_ip",
        ),
        PhaseConfig(
            name="phase25b_multiclass_groupaware_train_test",
            description="Phase 2.5b: Strict train/test source_ip-aware robustness check for the multiclass setup.",
            relative_files=[
                "01-12/DrDoS_DNS.csv",
                "01-12/DrDoS_NTP.csv",
                "01-12/DrDoS_NetBIOS.csv",
                "03-11/Portmap.csv",
            ],
            max_rows_total=200_000,
            target_mode="multiclass",
            sequence_lengths=[5, 10],
            lstm_hidden_sizes=[64],
            learning_rates=[1e-3],
            batch_sizes=[512],
            dropout_values=[0.2, 0.35],
            oversample_training=False,
            split_strategy="group_aware_train_test",
            group_column="source_ip",
            test_ratio=0.20,
        ),
    ]


def read_preview(path: Path, nrows: int = 5) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def load_phase_data(base_dir: Path, phase: PhaseConfig, chunksize: int = DEFAULT_CHUNK_SIZE) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    per_file_limit = max(1, phase.max_rows_total // max(1, len(phase.relative_files)))
    for relative_path in phase.relative_files:
        path = base_dir / relative_path
        collected = 0
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            remaining = per_file_limit - collected
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining].copy()
            chunk["__source_file__"] = relative_path
            frames.append(chunk)
            collected += len(chunk)
    if not frames:
        raise ValueError(f"No data loaded for phase {phase.name}.")
    return pd.concat(frames, ignore_index=True)


def summarize_raw_dataframe(df: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
    raw_target = infer_target_column(df.columns)
    summary = {
        "shape": list(df.shape),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024**2), 2),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "head": df.head(5).astype(str).to_dict(orient="records"),
        "target_column": raw_target,
        "target_distribution": (
            df[raw_target].astype(str).str.strip().value_counts(dropna=False).head(25).to_dict()
        ),
    }
    with (output_dir / "raw_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    return summary


def convert_numeric_like_columns(df: pd.DataFrame, exclude: Sequence[str]) -> Tuple[pd.DataFrame, List[str]]:
    converted: List[str] = []
    for column in df.columns:
        if column in exclude:
            continue
        if df[column].dtype != object:
            continue
        numeric_series = pd.to_numeric(df[column], errors="coerce")
        if numeric_series.notna().mean() >= 0.98:
            df[column] = numeric_series
            converted.append(column)
    return df, converted


def clean_labels(series: pd.Series, target_mode: str) -> pd.Series:
    labels = series.astype(str).str.strip()
    if target_mode == "binary":
        return labels.apply(lambda x: "ATTACK" if x.upper() != "BENIGN" else "BENIGN")
    return labels


def preprocess_dataframe(df: pd.DataFrame, phase: PhaseConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    metadata: Dict[str, Any] = {}
    original_columns = list(df.columns)
    df = df.copy()
    df.columns = standardize_columns(df.columns)
    metadata["standardized_columns"] = dict(zip(original_columns, df.columns))

    target_column = infer_target_column(df.columns)
    metadata["target_column"] = target_column

    for column in df.select_dtypes(include=["object"]).columns:
        df[column] = df[column].astype(str).str.strip()
        df.loc[df[column].isin(["", "nan", "None", "null", "NaN"]), column] = np.nan

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    exclude_from_numeric = {target_column, "__source_file__", "timestamp"}
    df, converted_columns = convert_numeric_like_columns(df, exclude_from_numeric)
    metadata["converted_numeric_columns"] = converted_columns

    metadata["duplicate_rows_removed"] = int(df.duplicated().sum())
    metadata["duplicate_columns_removed"] = find_duplicate_columns(df)
    df = df.drop_duplicates().drop(columns=metadata["duplicate_columns_removed"], errors="ignore")
    df = df.replace([np.inf, -np.inf], np.nan)

    missing_counts = df.isna().sum().sort_values(ascending=False)
    metadata["missing_values_top20"] = missing_counts.head(20).to_dict()

    constant_columns = [
        column for column in df.columns if column != target_column and df[column].nunique(dropna=False) <= 1
    ]
    near_constant_columns = []
    for column in df.columns:
        if column == target_column:
            continue
        value_frequencies = df[column].value_counts(dropna=False, normalize=True)
        if not value_frequencies.empty and value_frequencies.iloc[0] >= 0.995:
            near_constant_columns.append(column)
    metadata["constant_columns_removed"] = constant_columns
    metadata["near_constant_columns_removed"] = near_constant_columns

    metadata_only_columns = {"source_ip", "timestamp", "__source_file__"}
    high_cardinality_columns = []
    for column in df.select_dtypes(include=["object"]).columns:
        if column in {target_column, "__source_file__", "source_ip"}:
            continue
        if df[column].nunique(dropna=False) >= max(50, int(len(df) * 0.2)):
            high_cardinality_columns.append(column)
    metadata["high_cardinality_columns_removed"] = high_cardinality_columns

    leakage_reasons = identify_leakage_columns(df.columns)
    metadata["leakage_columns_removed"] = leakage_reasons

    columns_to_drop = sorted(
        set(constant_columns)
        | set(near_constant_columns)
        | set(high_cardinality_columns)
        | (set(leakage_reasons.keys()) - metadata_only_columns)
    )
    df = df.drop(columns=columns_to_drop, errors="ignore")

    df[target_column] = clean_labels(df[target_column], phase.target_mode)
    label_counts = df[target_column].value_counts().to_dict()
    metadata["label_distribution_after_cleaning"] = label_counts
    metadata["task_type"] = "binary" if len(label_counts) == 2 else "multiclass"
    metadata["final_shape"] = list(df.shape)
    metadata["final_columns"] = list(df.columns)
    return df, metadata


def save_json(data: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, default=str)


def plot_class_distribution(series: pd.Series, path: Path, title: str) -> None:
    counts = series.value_counts().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=counts.index.astype(str), y=counts.values, palette="viridis")
    plt.xticks(rotation=35, ha="right")
    plt.title(title)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()


def plot_missing_values(df: pd.DataFrame, path: Path) -> None:
    missing = df.isna().mean().sort_values(ascending=False).head(20)
    if missing.empty:
        return
    plt.figure(figsize=(10, 5))
    sns.barplot(x=missing.values, y=missing.index, palette="mako")
    plt.title("Top Missing Value Ratios")
    plt.xlabel("Missing Ratio")
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()


def plot_numeric_distributions(df: pd.DataFrame, features: Sequence[str], path: Path) -> None:
    selected = [feature for feature in features if feature in df.columns][:6]
    if not selected:
        return
    plt.figure(figsize=(14, 8))
    for idx, feature in enumerate(selected, start=1):
        plt.subplot(2, 3, idx)
        sns.histplot(df[feature].dropna(), bins=50, kde=False, color="#2a9d8f")
        plt.title(feature)
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, path: Path) -> List[Tuple[str, str, float]]:
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if numeric_df.shape[1] < 2:
        return []
    selected_cols = numeric_df.var().sort_values(ascending=False).head(MAX_CORR_FEATURES).index
    corr = numeric_df[selected_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr_pairs = []
    for col in upper.columns:
        for idx, value in upper[col].dropna().items():
            if value >= 0.98:
                high_corr_pairs.append((idx, col, float(value)))
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="rocket_r")
    plt.title("Correlation Heatmap of High-Variance Numeric Features")
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()
    return sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)[:30]


def stratified_split(
    df: pd.DataFrame,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        stratify=df[target_column],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df[target_column],
    )
    return train_df.copy(), val_df.copy(), test_df.copy()


def _summarize_split_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    group_column: Optional[str] = None,
) -> Dict[str, Any]:
    splits = {"train": train_df, "validation": val_df, "test": test_df}
    summary: Dict[str, Any] = {
        "row_counts": {name: int(len(frame)) for name, frame in splits.items()},
        "class_distribution": {
            name: frame[target_column].value_counts().to_dict() for name, frame in splits.items()
        },
    }
    if group_column:
        groups = {
            name: set(frame[group_column].dropna().astype(str).tolist()) for name, frame in splits.items()
        }
        summary["unique_group_counts"] = {name: len(group_set) for name, group_set in groups.items()}
        summary["group_overlap"] = {
            "train_validation": len(groups["train"] & groups["validation"]),
            "train_test": len(groups["train"] & groups["test"]),
            "validation_test": len(groups["validation"] & groups["test"]),
        }
    return summary


def assess_group_split_feasibility(
    df: pd.DataFrame,
    target_column: str,
    group_column: str,
    num_splits: int = 3,
) -> Dict[str, Any]:
    valid_df = df.dropna(subset=[group_column]).copy()
    groups_per_class = valid_df.groupby(target_column)[group_column].nunique().to_dict()
    infeasible_classes = {
        label: int(count) for label, count in groups_per_class.items() if int(count) < num_splits
    }
    return {
        "group_column": group_column,
        "num_splits_required": num_splits,
        "total_rows_after_group_na_drop": int(len(valid_df)),
        "unique_groups_total": int(valid_df[group_column].nunique()),
        "unique_groups_per_class": {label: int(count) for label, count in groups_per_class.items()},
        "is_feasible": len(infeasible_classes) == 0,
        "infeasible_classes": infeasible_classes,
    }


def group_aware_split(
    df: pd.DataFrame,
    target_column: str,
    group_column: str,
    seed: int = SEED,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    n_attempts: int = 80,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if group_column not in df.columns:
        raise ValueError(f"Group-aware split requested but '{group_column}' is missing.")
    if df[group_column].isna().all():
        raise ValueError(f"Group-aware split requested but '{group_column}' is entirely missing.")

    valid_df = df.dropna(subset=[group_column]).copy()
    classes = sorted(valid_df[target_column].astype(str).unique().tolist())
    split_names = ["train", "validation", "test"]
    feasibility = assess_group_split_feasibility(df, target_column, group_column, num_splits=len(split_names))
    if not feasibility["is_feasible"]:
        raise ValueError(
            "Strict group-aware splitting is infeasible for this dataset slice because some classes appear in "
            f"fewer than {len(split_names)} unique {group_column} groups: {feasibility['infeasible_classes']}."
        )
    split_ratios = {"train": train_ratio, "validation": val_ratio, "test": test_ratio}
    class_order = {label: idx for idx, label in enumerate(classes)}

    group_stats = valid_df.groupby(group_column).agg(
        row_count=(target_column, "size"),
        labels=(target_column, lambda s: s.tolist()),
    )
    group_records = []
    for group_name, row in group_stats.iterrows():
        label_counts = Counter(row["labels"])
        counts_vector = np.array([label_counts.get(label, 0) for label in classes], dtype=float)
        dominant_label = max(label_counts, key=label_counts.get)
        group_records.append(
            {
                "group": str(group_name),
                "row_count": int(row["row_count"]),
                "counts": counts_vector,
                "dominant_label": dominant_label,
            }
        )

    total_rows = float(len(valid_df))
    total_class_counts = np.sum([record["counts"] for record in group_records], axis=0)
    target_rows = {name: split_ratios[name] * total_rows for name in split_names}
    target_class_counts = {name: split_ratios[name] * total_class_counts for name in split_names}

    best_assignment: Optional[Dict[str, str]] = None
    best_score = math.inf
    rng_master = np.random.RandomState(seed)

    def score_assignment(rows_by_split: Dict[str, float], class_by_split: Dict[str, np.ndarray]) -> float:
        row_error = sum(
            abs(rows_by_split[name] - target_rows[name]) / max(1.0, target_rows[name]) for name in split_names
        )
        class_error = 0.0
        for name in split_names:
            class_error += float(
                np.sum(np.abs(class_by_split[name] - target_class_counts[name]) / np.maximum(target_class_counts[name], 1.0))
            )
        coverage_penalty = 0.0
        for name in split_names:
            present_classes = int(np.sum(class_by_split[name] > 0))
            coverage_penalty += (len(classes) - present_classes) * 25.0
        return row_error + class_error + coverage_penalty

    for attempt in range(n_attempts):
        attempt_seed = int(rng_master.randint(0, 1_000_000))
        rng = np.random.RandomState(attempt_seed)
        shuffled = list(group_records)
        rng.shuffle(shuffled)
        shuffled.sort(key=lambda item: (item["counts"].max(), item["row_count"]), reverse=True)

        assignment: Dict[str, str] = {}
        rows_by_split = {name: 0.0 for name in split_names}
        class_by_split = {name: np.zeros(len(classes), dtype=float) for name in split_names}

        for record in shuffled:
            best_split = None
            best_local_score = math.inf
            for split_name in split_names:
                temp_rows = {name: rows_by_split[name] for name in split_names}
                temp_class = {name: class_by_split[name].copy() for name in split_names}
                temp_rows[split_name] += record["row_count"]
                temp_class[split_name] += record["counts"]
                local_score = score_assignment(temp_rows, temp_class)
                if local_score < best_local_score:
                    best_local_score = local_score
                    best_split = split_name
            assert best_split is not None
            assignment[record["group"]] = best_split
            rows_by_split[best_split] += record["row_count"]
            class_by_split[best_split] += record["counts"]

        final_score = score_assignment(rows_by_split, class_by_split)
        if final_score < best_score:
            best_score = final_score
            best_assignment = assignment

    if best_assignment is None:
        raise ValueError("Failed to construct a group-aware split.")

    split_masks = {
        split_name: valid_df[group_column].astype(str).map(best_assignment).eq(split_name)
        for split_name in split_names
    }
    train_df = valid_df.loc[split_masks["train"]].copy()
    val_df = valid_df.loc[split_masks["validation"]].copy()
    test_df = valid_df.loc[split_masks["test"]].copy()

    for split_name, frame in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        missing_classes = sorted(set(classes) - set(frame[target_column].astype(str).unique().tolist()))
        if missing_classes:
            raise ValueError(
                f"Group-aware split left split '{split_name}' without classes: {missing_classes}. "
                "Strict grouping is possible here, but this assignment is not acceptable."
            )

    split_summary = _summarize_split_distribution(train_df, val_df, test_df, target_column, group_column)
    split_summary["split_strategy"] = "group_aware"
    split_summary["group_column"] = group_column
    split_summary["dropped_rows_missing_group"] = int(len(df) - len(valid_df))
    split_summary["balance_tradeoff_note"] = (
        "This split enforces zero source_ip overlap across splits. "
        "Because grouping is stricter than row-wise stratification, class balance is approximate rather than perfect."
    )
    if any(value != 0 for value in split_summary["group_overlap"].values()):
        raise ValueError("Group-aware split failed: source_ip overlap is not zero.")
    return train_df, val_df, test_df, split_summary


def group_aware_train_test_split(
    df: pd.DataFrame,
    target_column: str,
    group_column: str,
    test_ratio: float = 0.20,
    seed: int = SEED,
    n_attempts: int = 120,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if group_column not in df.columns:
        raise ValueError(f"Group-aware split requested but '{group_column}' is missing.")
    valid_df = df.dropna(subset=[group_column]).copy()
    classes = sorted(valid_df[target_column].astype(str).unique().tolist())
    feasibility = assess_group_split_feasibility(df, target_column, group_column, num_splits=2)
    if not feasibility["is_feasible"]:
        raise ValueError(
            "Strict group-aware train/test splitting is infeasible because some classes appear in fewer than "
            f"2 unique {group_column} groups: {feasibility['infeasible_classes']}."
        )

    group_class_table = valid_df.groupby(group_column)[target_column].value_counts().unstack(fill_value=0)
    for label in classes:
        if label not in group_class_table.columns:
            group_class_table[label] = 0
    group_class_table = group_class_table[classes]
    group_class_table["row_count"] = group_class_table.sum(axis=1)

    total_rows = float(len(valid_df))
    target_test_rows = total_rows * test_ratio
    total_class_counts = group_class_table[classes].sum(axis=0)
    target_test_class_counts = total_class_counts * test_ratio

    required_test_groups = set()
    for label in classes:
        positive_groups = group_class_table.index[group_class_table[label] > 0].tolist()
        if len(positive_groups) == 1:
            raise ValueError(
                f"Class '{label}' appears in only one {group_column} group after cleaning; strict train/test split is impossible."
            )
        required_test_groups.add(sorted(positive_groups, key=lambda g: str(g))[-1])

    remaining_groups = [g for g in group_class_table.index.tolist() if g not in required_test_groups]
    rng_master = np.random.RandomState(seed)
    best_test_groups: Optional[set[str]] = None
    best_score = math.inf

    def score_candidate(test_groups: set[str]) -> float:
        train_groups = set(group_class_table.index.tolist()) - test_groups
        if not train_groups or not test_groups:
            return math.inf
        test_counts = group_class_table.loc[list(test_groups), classes].sum(axis=0)
        train_counts = group_class_table.loc[list(train_groups), classes].sum(axis=0)
        if (test_counts <= 0).any() or (train_counts <= 0).any():
            return math.inf
        test_rows = float(group_class_table.loc[list(test_groups), "row_count"].sum())
        row_error = abs(test_rows - target_test_rows) / max(1.0, target_test_rows)
        class_error = float(
            np.sum(np.abs(test_counts.values - target_test_class_counts.values) / np.maximum(target_test_class_counts.values, 1.0))
        )
        fragility_penalty = float(np.sum((test_counts.values < 50) * 5.0))
        return row_error + class_error + fragility_penalty

    required_test_groups_str = {str(x) for x in required_test_groups}
    group_class_table.index = group_class_table.index.astype(str)
    remaining_groups = [str(g) for g in remaining_groups]

    for attempt in range(n_attempts):
        test_groups = set(required_test_groups_str)
        shuffled = list(remaining_groups)
        rng = np.random.RandomState(int(rng_master.randint(0, 1_000_000)))
        rng.shuffle(shuffled)
        for group in shuffled:
            current_score = score_candidate(test_groups)
            candidate_score = score_candidate(test_groups | {group})
            if candidate_score < current_score:
                test_groups.add(group)
        candidate_score = score_candidate(test_groups)
        if candidate_score < best_score:
            best_score = candidate_score
            best_test_groups = set(test_groups)

    if best_test_groups is None:
        raise ValueError("Failed to construct a group-aware train/test split.")

    test_mask = valid_df[group_column].astype(str).isin(best_test_groups)
    test_df = valid_df.loc[test_mask].copy()
    train_df = valid_df.loc[~test_mask].copy()

    split_summary = {
        "split_strategy": "group_aware_train_test",
        "group_column": group_column,
        "test_ratio_requested": test_ratio,
        "dropped_rows_missing_group": int(len(df) - len(valid_df)),
        "row_counts": {
            "train": int(len(train_df)),
            "test": int(len(test_df)),
        },
        "class_distribution": {
            "train": train_df[target_column].value_counts().to_dict(),
            "test": test_df[target_column].value_counts().to_dict(),
        },
        "unique_group_counts": {
            "train": int(train_df[group_column].astype(str).nunique()),
            "test": int(test_df[group_column].astype(str).nunique()),
        },
        "group_overlap": {
            "train_test": int(
                len(set(train_df[group_column].astype(str).unique()) & set(test_df[group_column].astype(str).unique()))
            ),
        },
        "unique_groups_per_class": feasibility["unique_groups_per_class"],
        "test_groups_with_attack_coverage": sorted(best_test_groups),
        "balance_tradeoff_note": (
            "Strict source_ip separation is enforced. Because each attack class has only two unique source_ip values, "
            "the resulting train/test class balance is inherently uneven and should be interpreted as a robustness check."
        ),
    }
    if split_summary["group_overlap"]["train_test"] != 0:
        raise ValueError("Group-aware train/test split failed: source_ip overlap is not zero.")
    return train_df, test_df, split_summary


def stratified_validation_from_train(
    train_df: pd.DataFrame,
    target_column: str,
    validation_ratio: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    inner_train_df, val_df = train_test_split(
        train_df,
        test_size=validation_ratio,
        random_state=SEED,
        stratify=train_df[target_column],
    )
    summary = {
        "strategy": "row_stratified_within_train_only",
        "row_counts": {
            "inner_train": int(len(inner_train_df)),
            "validation": int(len(val_df)),
        },
        "class_distribution": {
            "inner_train": inner_train_df[target_column].value_counts().to_dict(),
            "validation": val_df[target_column].value_counts().to_dict(),
        },
        "note": (
            "This validation split is used only for model selection inside the training partition. "
            "The strict external robustness check remains the source_ip-aware train/test split."
        ),
    }
    return inner_train_df.copy(), val_df.copy(), summary


def build_preprocessor(feature_columns: Sequence[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(transformers=[("numeric", numeric_pipeline, list(feature_columns))])


def prepare_model_inputs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    output_dir: Path,
    phase: PhaseConfig,
) -> Dict[str, Any]:
    helper_columns = [target_column, "__source_file__", "source_ip", "timestamp", "destination_ip"]
    feature_columns = [
        col
        for col in train_df.columns
        if col not in helper_columns and pd.api.types.is_numeric_dtype(train_df[col])
    ]
    preprocessor = build_preprocessor(feature_columns)

    X_train = preprocessor.fit_transform(train_df[feature_columns])
    X_val = preprocessor.transform(val_df[feature_columns])
    X_test = preprocessor.transform(test_df[feature_columns])

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df[target_column])
    y_val = label_encoder.transform(val_df[target_column])
    y_test = label_encoder.transform(test_df[target_column])

    class_counts = Counter(y_train.tolist())
    num_classes = len(label_encoder.classes_)
    total = len(y_train)
    class_weights = {class_index: total / (num_classes * count) for class_index, count in class_counts.items()}

    if phase.oversample_training:
        sampler = RandomOverSampler(random_state=SEED)
        X_train, y_train = sampler.fit_resample(X_train, y_train)

    save_json(
        {
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "classes": label_encoder.classes_.tolist(),
            "class_weights": class_weights,
        },
        output_dir / "preprocessing_summary.json",
    )
    return {
        "feature_columns": feature_columns,
        "preprocessor": preprocessor,
        "label_encoder": label_encoder,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "class_weights": class_weights,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }


def compute_candidate_feature_importance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_columns: Sequence[str],
    output_dir: Path,
) -> Dict[str, float]:
    sample_size = min(40_000, len(y_train))
    indices = np.random.RandomState(SEED).choice(len(y_train), size=sample_size, replace=False)
    model = ExtraTreesClassifier(
        n_estimators=150,
        random_state=SEED,
        n_jobs=1,
        class_weight="balanced_subsample",
    )
    model.fit(X_train[indices], y_train[indices])
    pairs = sorted(zip(feature_columns, model.feature_importances_), key=lambda x: x[1], reverse=True)[:20]
    importance_dict = {name: round(float(score), 6) for name, score in pairs}

    plt.figure(figsize=(10, 6))
    sns.barplot(x=[score for _, score in pairs][::-1], y=[name for name, _ in pairs][::-1], palette="crest")
    plt.title("Top Candidate Informative Features")
    plt.tight_layout()
    plt.savefig(output_dir / "candidate_feature_importance.png", dpi=PLOT_DPI)
    plt.close()
    return importance_dict


def train_logistic_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: Dict[int, float],
) -> Tuple[LogisticRegression, Dict[str, Any]]:
    candidates = [0.3, 1.0, 3.0]
    best_model: Optional[LogisticRegression] = None
    best_metrics: Optional[Dict[str, Any]] = None
    for c_value in candidates:
        model = LogisticRegression(
            C=c_value,
            max_iter=1200,
            solver="saga",
            n_jobs=1,
            class_weight=class_weights,
        )
        model.fit(X_train, y_train)
        val_predictions = model.predict(X_val)
        val_f1 = f1_score(y_val, val_predictions, average="weighted")
        metrics = {"C": c_value, "validation_weighted_f1": float(val_f1)}
        if best_metrics is None or metrics["validation_weighted_f1"] > best_metrics["validation_weighted_f1"]:
            best_model = model
            best_metrics = metrics
    assert best_model is not None and best_metrics is not None
    return best_model, best_metrics


def train_random_forest_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: Dict[int, float],
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    candidates = [
        {"n_estimators": 200, "max_depth": None},
        {"n_estimators": 300, "max_depth": 20},
    ]
    best_model: Optional[RandomForestClassifier] = None
    best_metrics: Optional[Dict[str, Any]] = None
    for params in candidates:
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=SEED,
            n_jobs=1,
            class_weight=class_weights,
        )
        model.fit(X_train, y_train)
        val_predictions = model.predict(X_val)
        val_f1 = f1_score(y_val, val_predictions, average="weighted")
        metrics = {
            "n_estimators": params["n_estimators"],
            "max_depth": params["max_depth"],
            "validation_weighted_f1": float(val_f1),
        }
        if best_metrics is None or metrics["validation_weighted_f1"] > best_metrics["validation_weighted_f1"]:
            best_model = model
            best_metrics = metrics
    assert best_model is not None and best_metrics is not None
    return best_model, best_metrics


def train_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: Dict[int, float],
) -> Tuple[Any, Dict[str, Any]]:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ValueError("xgboost is not installed.") from exc

    sample_weight_train = np.asarray([class_weights[int(label)] for label in y_train], dtype=float)
    sample_weight_val = np.asarray([class_weights[int(label)] for label in y_val], dtype=float)
    candidates = [
        {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1},
        {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05},
    ]
    best_model = None
    best_metrics = None
    num_classes = len(np.unique(y_train))
    for params in candidates:
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            random_state=SEED,
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            n_jobs=1,
        )
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight_train,
            eval_set=[(X_val, y_val)],
            sample_weight_eval_set=[sample_weight_val],
            verbose=False,
        )
        val_predictions = model.predict(X_val)
        val_f1 = f1_score(y_val, val_predictions, average="weighted")
        metrics = {
            "n_estimators": params["n_estimators"],
            "max_depth": params["max_depth"],
            "learning_rate": params["learning_rate"],
            "validation_weighted_f1": float(val_f1),
        }
        if best_metrics is None or metrics["validation_weighted_f1"] > best_metrics["validation_weighted_f1"]:
            best_model = model
            best_metrics = metrics
    assert best_model is not None and best_metrics is not None
    return best_model, best_metrics


def evaluate_classifier(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    class_names: Sequence[str],
) -> Dict[str, Any]:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics: Dict[str, Any] = {
        "model": model_name,
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "classification_report": classification_report(
            y_true, y_pred, target_names=list(class_names), zero_division=0, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_proba is not None:
        try:
            if len(class_names) == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
                metrics["pr_auc"] = float(average_precision_score(y_true, y_proba[:, 1]))
            else:
                y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
                metrics["roc_auc_ovr_weighted"] = float(
                    roc_auc_score(y_true_bin, y_proba, average="weighted", multi_class="ovr")
                )
                metrics["pr_auc_weighted"] = float(
                    average_precision_score(y_true_bin, y_proba, average="weighted")
                )
        except Exception as exc:
            metrics["probability_metric_error"] = str(exc)
    return metrics


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], path: Path, title: str) -> None:
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()


class FlowSequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray) -> None:
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[index], self.labels[index]


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        reduced_size = hidden_size // 2 if hidden_size >= 2 else 1
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, reduced_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(reduced_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_hidden = self.dropout(output[:, -1, :])
        return self.classifier(last_hidden)


def build_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    df_reference: pd.DataFrame,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    ref = df_reference.copy()
    ref["__row_pos__"] = np.arange(len(ref))
    if "timestamp" in ref.columns:
        ref["timestamp"] = pd.to_datetime(ref["timestamp"], errors="coerce")
    else:
        ref["timestamp"] = pd.Timestamp("1970-01-01")
    if "source_ip" not in ref.columns:
        ref["source_ip"] = ref.get("__source_file__", "global")
    ref = ref.sort_values(["source_ip", "timestamp", "__row_pos__"]).reset_index(drop=True)

    sequence_buffer: Dict[str, List[np.ndarray]] = defaultdict(list)
    sequences: List[np.ndarray] = []
    sequence_labels: List[int] = []
    for _, row in ref.iterrows():
        original_idx = int(row["__row_pos__"])
        group_key = str(row["source_ip"])
        history = sequence_buffer[group_key][-max(sequence_length - 1, 0) :]
        current = features[original_idx]
        full_sequence = history + [current]
        if len(full_sequence) < sequence_length:
            padding = [np.zeros_like(current) for _ in range(sequence_length - len(full_sequence))]
            full_sequence = padding + full_sequence
        sequences.append(np.stack(full_sequence[-sequence_length:], axis=0))
        sequence_labels.append(int(labels[original_idx]))
        sequence_buffer[group_key].append(current)
    return np.asarray(sequences, dtype=np.float32), np.asarray(sequence_labels, dtype=np.int64)


def create_data_loader(sequences: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(FlowSequenceDataset(sequences, labels), batch_size=batch_size, shuffle=shuffle)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> Tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    all_predictions: List[int] = []
    all_labels: List[int] = []
    with torch.set_grad_enabled(is_training):
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(batch_labels)
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.detach().cpu().numpy().tolist())
            all_labels.extend(batch_labels.detach().cpu().numpy().tolist())
    avg_loss = total_loss / max(1, len(loader.dataset))
    weighted_f1 = f1_score(all_labels, all_predictions, average="weighted", zero_division=0)
    return avg_loss, float(weighted_f1)


def predict_with_lstm(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: List[np.ndarray] = []
    predictions: List[int] = []
    with torch.no_grad():
        for batch_features, _ in loader:
            logits = model(batch_features.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probabilities.append(probs)
            predictions.extend(np.argmax(probs, axis=1).tolist())
    return np.asarray(predictions), np.vstack(probabilities)


def train_single_lstm_config(
    prepared: Dict[str, Any],
    sequence_length: int,
    hidden_size: int,
    dropout: float,
    learning_rate: float,
    batch_size: int,
    max_epochs: int = 10,
    patience: int = 2,
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, List[float]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights_tensor = torch.tensor(
        [prepared["class_weights"][idx] for idx in range(len(prepared["label_encoder"].classes_))],
        dtype=torch.float32,
        device=device,
    )

    X_train_seq, y_train_seq = build_sequences(prepared["X_train"], prepared["y_train"], prepared["train_df"], sequence_length)
    X_val_seq, y_val_seq = build_sequences(prepared["X_val"], prepared["y_val"], prepared["val_df"], sequence_length)
    train_loader = create_data_loader(X_train_seq, y_train_seq, batch_size, True)
    val_loader = create_data_loader(X_val_seq, y_val_seq, batch_size, False)

    model = LSTMClassifier(
        input_size=X_train_seq.shape[2],
        hidden_size=hidden_size,
        output_size=len(prepared["label_encoder"].classes_),
        dropout=dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    best_val_for_run = -math.inf
    best_run_weights = None
    patience_counter = 0
    for _ in range(max_epochs):
        train_loss, train_f1 = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1 = run_epoch(model, val_loader, criterion, None, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)
        if val_f1 > best_val_for_run:
            best_val_for_run = val_f1
            best_run_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break
    if best_run_weights is not None:
        model.load_state_dict(best_run_weights)
    config = {
        "sequence_length": sequence_length,
        "hidden_size": hidden_size,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "best_validation_weighted_f1": float(best_val_for_run),
        "epochs_ran": len(history["val_f1"]),
    }
    return model, config, history


def train_lstm_with_tuning(
    phase: PhaseConfig,
    prepared: Dict[str, Any],
    output_dir: Path,
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, List[float]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights_tensor = torch.tensor(
        [prepared["class_weights"][idx] for idx in range(len(prepared["label_encoder"].classes_))],
        dtype=torch.float32,
        device=device,
    )

    best_model: Optional[nn.Module] = None
    best_state: Optional[Dict[str, Any]] = None
    best_history: Optional[Dict[str, List[float]]] = None
    best_score = -math.inf
    tuning_results: List[Dict[str, Any]] = []

    for sequence_length in phase.sequence_lengths:
        X_train_seq, y_train_seq = build_sequences(prepared["X_train"], prepared["y_train"], prepared["train_df"], sequence_length)
        X_val_seq, y_val_seq = build_sequences(prepared["X_val"], prepared["y_val"], prepared["val_df"], sequence_length)
        for hidden_size in phase.lstm_hidden_sizes:
            for dropout in phase.dropout_values:
                for learning_rate in phase.learning_rates:
                    for batch_size in phase.batch_sizes:
                        train_loader = create_data_loader(X_train_seq, y_train_seq, batch_size, True)
                        val_loader = create_data_loader(X_val_seq, y_val_seq, batch_size, False)
                        model = LSTMClassifier(
                            input_size=X_train_seq.shape[2],
                            hidden_size=hidden_size,
                            output_size=len(prepared["label_encoder"].classes_),
                            dropout=dropout,
                        ).to(device)
                        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
                        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
                        history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
                        best_val_for_run = -math.inf
                        patience = 3
                        patience_counter = 0
                        best_run_weights = None
                        for _ in range(12):
                            train_loss, train_f1 = run_epoch(model, train_loader, criterion, optimizer, device)
                            val_loss, val_f1 = run_epoch(model, val_loader, criterion, None, device)
                            history["train_loss"].append(train_loss)
                            history["val_loss"].append(val_loss)
                            history["train_f1"].append(train_f1)
                            history["val_f1"].append(val_f1)
                            if val_f1 > best_val_for_run:
                                best_val_for_run = val_f1
                                patience_counter = 0
                                best_run_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                            else:
                                patience_counter += 1
                            if patience_counter >= patience:
                                break
                        if best_run_weights is not None:
                            model.load_state_dict(best_run_weights)
                        result = {
                            "sequence_length": sequence_length,
                            "hidden_size": hidden_size,
                            "dropout": dropout,
                            "learning_rate": learning_rate,
                            "batch_size": batch_size,
                            "best_validation_weighted_f1": float(best_val_for_run),
                            "epochs_ran": len(history["val_f1"]),
                        }
                        tuning_results.append(result)
                        if best_val_for_run > best_score:
                            best_score = best_val_for_run
                            best_model = model
                            best_state = result
                            best_history = history

    assert best_model is not None and best_state is not None and best_history is not None
    save_json({"results": tuning_results, "best": best_state}, output_dir / "lstm_tuning_results.json")
    return best_model, best_state, best_history


def plot_training_history(history: Dict[str, List[float]], path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Loss Curves")
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_f1"], label="Train Weighted F1")
    plt.plot(epochs, history["val_f1"], label="Val Weighted F1")
    plt.legend()
    plt.title("Weighted F1 Curves")
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()


def build_phase3_prepared(base_dir: Path, output_dir: Path) -> Tuple[PhaseConfig, pd.DataFrame, Dict[str, Any], Dict[str, Any], Path]:
    phase = next(cfg for cfg in default_phase_configs() if cfg.name == "phase25b_multiclass_groupaware_train_test")
    phase = PhaseConfig(**{**phase.__dict__, "max_rows_total": 60_000})
    ensure_dir(output_dir)
    raw_df = load_phase_data(base_dir, phase)
    raw_summary = summarize_raw_dataframe(raw_df, output_dir)
    cleaned_df, cleaning = preprocess_dataframe(raw_df, phase)
    save_json(cleaning, output_dir / "cleaning_summary.json")
    train_df, test_df, split_summary = group_aware_train_test_split(
        cleaned_df,
        cleaning["target_column"],
        phase.group_column or "source_ip",
        test_ratio=phase.test_ratio,
        seed=SEED,
    )
    inner_train_df, val_df, internal_validation_summary = stratified_validation_from_train(train_df, cleaning["target_column"])
    split_summary["internal_validation"] = internal_validation_summary
    save_json(split_summary, output_dir / "split_summary.json")
    prepared = prepare_model_inputs(inner_train_df, val_df, test_df, cleaning["target_column"], output_dir, phase)
    return phase, cleaned_df, {"raw_summary": raw_summary, "cleaning": cleaning, "split_summary": split_summary}, prepared, output_dir


def evaluate_trained_model(
    model_name: str,
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Sequence[str],
) -> Dict[str, Any]:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    return evaluate_classifier(model_name, y_test, predictions, probabilities, class_names)


def get_top_dominant_features(feature_importance: Dict[str, float], exclude: Optional[Sequence[str]] = None, top_k: int = 3) -> List[str]:
    exclude_set = set(exclude or [])
    ranked = [feature for feature, _ in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)]
    return [feature for feature in ranked if feature not in exclude_set][:top_k]


def prepare_model_inputs_with_feature_subset(
    prepared_source: Dict[str, Any],
    target_column: str,
    output_dir: Path,
    phase: PhaseConfig,
    dropped_features: Sequence[str],
) -> Dict[str, Any]:
    train_df = prepared_source["train_df"].drop(columns=list(dropped_features), errors="ignore")
    val_df = prepared_source["val_df"].drop(columns=list(dropped_features), errors="ignore")
    test_df = prepared_source["test_df"].drop(columns=list(dropped_features), errors="ignore")
    return prepare_model_inputs(train_df, val_df, test_df, target_column, output_dir, phase)


def run_phase3_generalization(base_dir: Path, output_root: Path) -> Dict[str, Any]:
    phase, cleaned_df, context, prepared, output_dir = build_phase3_prepared(base_dir, output_root)
    raw_summary = context["raw_summary"]
    cleaning = context["cleaning"]
    split_summary = context["split_summary"]
    target_column = cleaning["target_column"]
    class_names = prepared["label_encoder"].classes_

    candidate_feature_importance = compute_candidate_feature_importance(
        prepared["X_train"], prepared["y_train"], prepared["feature_columns"], output_dir
    )
    save_json(candidate_feature_importance, output_dir / "candidate_feature_importance.json")

    dominant_without_inbound = get_top_dominant_features(candidate_feature_importance, exclude=["inbound"], top_k=1)
    ablation_specs = [
        {"name": "full_features", "drop_features": []},
        {"name": "without_inbound", "drop_features": ["inbound"]},
        {"name": "without_top_dominant", "drop_features": dominant_without_inbound},
    ]

    feature_ablation_results: List[Dict[str, Any]] = []
    for spec in ablation_specs:
        ablation_output_dir = output_dir / f"ablation_{spec['name']}"
        ensure_dir(ablation_output_dir)
        prepared_subset = prepare_model_inputs_with_feature_subset(
            prepared,
            target_column,
            ablation_output_dir,
            phase,
            spec["drop_features"],
        )
        baseline_model, baseline_tuning = train_logistic_baseline(
            prepared_subset["X_train"],
            prepared_subset["y_train"],
            prepared_subset["X_val"],
            prepared_subset["y_val"],
            prepared_subset["class_weights"],
        )
        baseline_metrics = evaluate_trained_model(
            f"logistic_regression_{spec['name']}",
            baseline_model,
            prepared_subset["X_test"],
            prepared_subset["y_test"],
            prepared_subset["label_encoder"].classes_,
        )
        baseline_metrics["tuning"] = baseline_tuning
        feature_ablation_results.append(
            {
                "experiment": spec["name"],
                "dropped_features": spec["drop_features"],
                "baseline_accuracy": baseline_metrics["accuracy"],
                "baseline_weighted_f1": baseline_metrics["f1_weighted"],
                "baseline_macro_f1": baseline_metrics["f1_macro"],
            }
        )
    save_json({"dominant_features": dominant_without_inbound, "results": feature_ablation_results}, output_dir / "feature_ablation_results.json")

    model_comparison_results: Dict[str, Any] = {}
    baseline_model, baseline_tuning = train_logistic_baseline(
        prepared["X_train"], prepared["y_train"], prepared["X_val"], prepared["y_val"], prepared["class_weights"]
    )
    baseline_metrics = evaluate_trained_model(
        "logistic_regression", baseline_model, prepared["X_test"], prepared["y_test"], class_names
    )
    baseline_metrics["tuning"] = baseline_tuning
    plot_confusion_matrix(
        np.array(baseline_metrics["confusion_matrix"]),
        class_names,
        output_dir / "baseline_confusion_matrix.png",
        "Phase 3 Logistic Regression Confusion Matrix",
    )
    model_comparison_results["logistic_regression"] = baseline_metrics

    random_forest_model, random_forest_tuning = train_random_forest_baseline(
        prepared["X_train"], prepared["y_train"], prepared["X_val"], prepared["y_val"], prepared["class_weights"]
    )
    random_forest_metrics = evaluate_trained_model(
        "random_forest", random_forest_model, prepared["X_test"], prepared["y_test"], class_names
    )
    random_forest_metrics["tuning"] = random_forest_tuning
    plot_confusion_matrix(
        np.array(random_forest_metrics["confusion_matrix"]),
        class_names,
        output_dir / "random_forest_confusion_matrix.png",
        "Phase 3 Random Forest Confusion Matrix",
    )
    model_comparison_results["random_forest"] = random_forest_metrics

    sequence_results: List[Dict[str, Any]] = []
    best_lstm_metrics = None
    best_lstm_history = None
    best_lstm_config = None
    for sequence_length in [5, 10]:
        lstm_model, lstm_config, lstm_history = train_single_lstm_config(
            prepared,
            sequence_length=sequence_length,
            hidden_size=64,
            dropout=0.2,
            learning_rate=1e-3,
            batch_size=512,
        )
        X_test_seq, y_test_seq = build_sequences(
            prepared["X_test"], prepared["y_test"], prepared["test_df"], int(lstm_config["sequence_length"])
        )
        test_loader = create_data_loader(X_test_seq, y_test_seq, int(lstm_config["batch_size"]), False)
        lstm_pred, lstm_proba = predict_with_lstm(
            lstm_model, test_loader, torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        lstm_metrics = evaluate_classifier("lstm", y_test_seq, lstm_pred, lstm_proba, class_names)
        result = {
            "sequence_length": sequence_length,
            "best_validation_weighted_f1": lstm_config["best_validation_weighted_f1"],
            "test_accuracy": lstm_metrics["accuracy"],
            "test_weighted_f1": lstm_metrics["f1_weighted"],
            "test_macro_f1": lstm_metrics["f1_macro"],
            "config": lstm_config,
        }
        sequence_results.append(result)
        if best_lstm_metrics is None or lstm_metrics["f1_weighted"] > best_lstm_metrics["f1_weighted"]:
            best_lstm_metrics = lstm_metrics
            best_lstm_history = lstm_history
            best_lstm_config = lstm_config

    assert best_lstm_metrics is not None and best_lstm_history is not None and best_lstm_config is not None
    best_lstm_metrics["best_config"] = best_lstm_config
    model_comparison_results["lstm"] = best_lstm_metrics
    save_json({"results": sequence_results, "best": best_lstm_config}, output_dir / "lstm_sequence_experiments.json")
    plot_training_history(best_lstm_history, output_dir / "lstm_training_curves.png")
    plot_confusion_matrix(
        np.array(best_lstm_metrics["confusion_matrix"]),
        class_names,
        output_dir / "lstm_confusion_matrix.png",
        "Phase 3 LSTM Confusion Matrix",
    )

    save_json(model_comparison_results, output_dir / "model_comparison_results.json")

    phase2_baseline = json.loads((base_dir / "outputs_phase2_clean" / "phase2_multiclass_subset" / "baseline_test_metrics.json").read_text(encoding="utf-8"))
    phase2_lstm = json.loads((base_dir / "outputs_phase2_clean" / "phase2_multiclass_subset" / "lstm_test_metrics.json").read_text(encoding="utf-8"))
    phase25b_baseline = json.loads((base_dir / "outputs_phase25b_groupaware_train_test" / "phase25b_multiclass_groupaware_train_test" / "baseline_test_metrics.json").read_text(encoding="utf-8"))
    phase25b_lstm = json.loads((base_dir / "outputs_phase25b_groupaware_train_test" / "phase25b_multiclass_groupaware_train_test" / "lstm_test_metrics.json").read_text(encoding="utf-8"))

    final_summary = {
        "phase3_protocol": "strict_group_aware_train_test_by_source_ip",
        "phase3_design": {
            "max_rows_total": phase.max_rows_total,
            "ablation_experiments": [spec["name"] for spec in ablation_specs],
            "ablation_model": "logistic_regression",
            "models_compared": ["logistic_regression", "random_forest", "lstm"],
            "lstm_sequence_lengths": [5, 10],
        },
        "raw_summary": raw_summary,
        "cleaning": cleaning,
        "split_summary": split_summary,
        "feature_importance": candidate_feature_importance,
        "feature_ablation_results": feature_ablation_results,
        "model_comparison": {
            model_name: {
                "accuracy": metrics["accuracy"],
                "f1_weighted": metrics["f1_weighted"],
                "f1_macro": metrics["f1_macro"],
            }
            for model_name, metrics in model_comparison_results.items()
        },
        "phase2_reference": {
            "baseline": {
                "accuracy": phase2_baseline["accuracy"],
                "f1_weighted": phase2_baseline["f1_weighted"],
                "f1_macro": phase2_baseline["classification_report"]["macro avg"]["f1-score"],
            },
            "lstm": {
                "accuracy": phase2_lstm["accuracy"],
                "f1_weighted": phase2_lstm["f1_weighted"],
                "f1_macro": phase2_lstm["classification_report"]["macro avg"]["f1-score"],
            },
        },
        "phase25b_reference": {
            "baseline": {
                "accuracy": phase25b_baseline["accuracy"],
                "f1_weighted": phase25b_baseline["f1_weighted"],
                "f1_macro": phase25b_baseline["f1_macro"],
            },
            "lstm": {
                "accuracy": phase25b_lstm["accuracy"],
                "f1_weighted": phase25b_lstm["f1_weighted"],
                "f1_macro": phase25b_lstm["f1_macro"],
            },
        },
    }
    save_json(final_summary, output_dir / "final_summary.json")
    return final_summary


def run_phase(base_dir: Path, phase: PhaseConfig, output_root: Path) -> Dict[str, Any]:
    phase_dir = output_root / phase.name
    ensure_dir(phase_dir)

    raw_df = load_phase_data(base_dir, phase)
    raw_summary = summarize_raw_dataframe(raw_df, phase_dir)

    cleaned_df, cleaning_metadata = preprocess_dataframe(raw_df, phase)
    target_column = cleaning_metadata["target_column"]
    save_json(cleaning_metadata, phase_dir / "cleaning_summary.json")

    plot_class_distribution(cleaned_df[target_column], phase_dir / "class_distribution.png", f"{phase.name} Class Distribution")
    plot_missing_values(cleaned_df, phase_dir / "missing_values.png")
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
    plot_numeric_distributions(cleaned_df, numeric_cols[:6], phase_dir / "numeric_distributions.png")
    high_corr_pairs = plot_correlation_heatmap(cleaned_df, phase_dir / "correlation_heatmap.png")
    save_json({"highly_correlated_pairs": high_corr_pairs}, phase_dir / "correlation_summary.json")

    internal_validation_summary = None
    if phase.split_strategy == "group_aware":
        if not phase.group_column:
            raise ValueError("Group-aware split strategy requires a group_column.")
        train_df, val_df, test_df, split_summary = group_aware_split(
            cleaned_df,
            target_column,
            phase.group_column,
            seed=SEED,
        )
    elif phase.split_strategy == "group_aware_train_test":
        if not phase.group_column:
            raise ValueError("Group-aware train/test split strategy requires a group_column.")
        train_df, test_df, split_summary = group_aware_train_test_split(
            cleaned_df,
            target_column,
            phase.group_column,
            test_ratio=phase.test_ratio,
            seed=SEED,
        )
        inner_train_df, val_df, internal_validation_summary = stratified_validation_from_train(train_df, target_column)
        split_summary["internal_validation"] = internal_validation_summary
        train_df = inner_train_df
    else:
        train_df, val_df, test_df = stratified_split(cleaned_df, target_column)
        split_summary = _summarize_split_distribution(train_df, val_df, test_df, target_column, phase.group_column)
        split_summary["split_strategy"] = "row_stratified"
        if phase.group_column:
            split_summary["group_column"] = phase.group_column
    save_json(split_summary, phase_dir / "split_summary.json")

    prepared = prepare_model_inputs(train_df, val_df, test_df, target_column, phase_dir, phase)

    candidate_feature_importance = compute_candidate_feature_importance(
        prepared["X_train"], prepared["y_train"], prepared["feature_columns"], phase_dir
    )
    save_json(candidate_feature_importance, phase_dir / "candidate_feature_importance.json")

    baseline_model, baseline_tuning = train_logistic_baseline(
        prepared["X_train"],
        prepared["y_train"],
        prepared["X_val"],
        prepared["y_val"],
        prepared["class_weights"],
    )
    baseline_pred = baseline_model.predict(prepared["X_test"])
    baseline_proba = baseline_model.predict_proba(prepared["X_test"])
    baseline_metrics = evaluate_classifier(
        "logistic_regression",
        prepared["y_test"],
        baseline_pred,
        baseline_proba,
        prepared["label_encoder"].classes_,
    )
    baseline_metrics["tuning"] = baseline_tuning
    save_json(baseline_metrics, phase_dir / "baseline_test_metrics.json")
    plot_confusion_matrix(
        np.array(baseline_metrics["confusion_matrix"]),
        prepared["label_encoder"].classes_,
        phase_dir / "baseline_confusion_matrix.png",
        "Baseline Confusion Matrix",
    )

    best_lstm_model, best_lstm_config, best_lstm_history = train_lstm_with_tuning(phase, prepared, phase_dir)
    plot_training_history(best_lstm_history, phase_dir / "lstm_training_curves.png")

    best_sequence_length = int(best_lstm_config["sequence_length"])
    X_test_seq, y_test_seq = build_sequences(prepared["X_test"], prepared["y_test"], prepared["test_df"], best_sequence_length)
    test_loader = create_data_loader(X_test_seq, y_test_seq, int(best_lstm_config["batch_size"]), False)
    lstm_pred, lstm_proba = predict_with_lstm(
        best_lstm_model,
        test_loader,
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    lstm_metrics = evaluate_classifier(
        "lstm",
        y_test_seq,
        lstm_pred,
        lstm_proba,
        prepared["label_encoder"].classes_,
    )
    lstm_metrics["best_config"] = best_lstm_config
    save_json(lstm_metrics, phase_dir / "lstm_test_metrics.json")
    plot_confusion_matrix(
        np.array(lstm_metrics["confusion_matrix"]),
        prepared["label_encoder"].classes_,
        phase_dir / "lstm_confusion_matrix.png",
        "LSTM Confusion Matrix",
    )

    comparison = {
        "phase_name": phase.name,
        "description": phase.description,
        "raw_summary": raw_summary,
        "cleaning": cleaning_metadata,
        "split_summary": split_summary,
        "baseline_metrics": baseline_metrics,
        "lstm_metrics": lstm_metrics,
        "candidate_feature_importance": candidate_feature_importance,
        "sequence_methodology": (
            "Rows are grouped by source_ip and ordered by timestamp within each split. "
            "Each LSTM sample uses the current flow plus up to sequence_length-1 prior flows from the same source_ip. "
            "This is a reasonable proxy for temporal behavior, but CIC-DDoS2019 is primarily a tabular flow dataset."
        ),
    }
    if phase.name in {"phase25_multiclass_groupaware", "phase25b_multiclass_groupaware_train_test"}:
        phase2_dir = base_dir / "outputs_phase2_clean" / "phase2_multiclass_subset"
        if (phase2_dir / "baseline_test_metrics.json").exists() and (phase2_dir / "lstm_test_metrics.json").exists():
            phase2_baseline = json.loads((phase2_dir / "baseline_test_metrics.json").read_text(encoding="utf-8"))
            phase2_lstm = json.loads((phase2_dir / "lstm_test_metrics.json").read_text(encoding="utf-8"))
            phase25_vs_phase2 = {
                "baseline_accuracy_delta": float(baseline_metrics["accuracy"] - phase2_baseline["accuracy"]),
                "baseline_weighted_f1_delta": float(baseline_metrics["f1_weighted"] - phase2_baseline["f1_weighted"]),
                "baseline_macro_f1_delta": float(
                    baseline_metrics["f1_macro"]
                    - phase2_baseline.get(
                        "f1_macro",
                        phase2_baseline.get("classification_report", {}).get("macro avg", {}).get("f1-score", 0.0),
                    )
                ),
                "lstm_accuracy_delta": float(lstm_metrics["accuracy"] - phase2_lstm["accuracy"]),
                "lstm_weighted_f1_delta": float(lstm_metrics["f1_weighted"] - phase2_lstm["f1_weighted"]),
                "lstm_macro_f1_delta": float(
                    lstm_metrics["f1_macro"]
                    - phase2_lstm.get(
                        "f1_macro",
                        phase2_lstm.get("classification_report", {}).get("macro avg", {}).get("f1-score", 0.0),
                    )
                ),
            }
            comparison["phase2_reference_metrics"] = {
                "baseline": {
                    "accuracy": phase2_baseline["accuracy"],
                    "f1_weighted": phase2_baseline["f1_weighted"],
                    "f1_macro": phase2_baseline.get(
                        "f1_macro",
                        phase2_baseline.get("classification_report", {}).get("macro avg", {}).get("f1-score"),
                    ),
                },
                "lstm": {
                    "accuracy": phase2_lstm["accuracy"],
                    "f1_weighted": phase2_lstm["f1_weighted"],
                    "f1_macro": phase2_lstm.get(
                        "f1_macro",
                        phase2_lstm.get("classification_report", {}).get("macro avg", {}).get("f1-score"),
                    ),
                },
            }
            comparison["phase25_vs_phase2"] = phase25_vs_phase2
            save_json(comparison["phase25_vs_phase2"], phase_dir / "phase2_comparison_summary.json")
    save_json(comparison, phase_dir / "phase_summary.json")
    return comparison


def generate_final_report(results: List[Dict[str, Any]], output_root: Path) -> None:
    lines = [
        "# Detecting DDoS Attacks using LSTM on the CIC-DDoS2019 dataset",
        "",
        "## Final concise summary",
        "",
    ]
    for result in results:
        phase = result["phase_name"]
        cleaning = result["cleaning"]
        baseline = result["baseline_metrics"]
        lstm = result["lstm_metrics"]
        lines.extend(
            [
                f"### {phase}",
                f"- Dataset quality assessment: final cleaned shape {cleaning['final_shape']} with task type `{cleaning['task_type']}`.",
                "- Main preprocessing decisions: dropped leakage-prone identifiers, removed constant or near-constant features, converted numeric-like object columns, deduplicated rows and columns, used median imputation plus standard scaling, and fit preprocessing on training data only.",
                f"- Best model comparison: baseline weighted F1 = {baseline['f1_weighted']:.4f}, LSTM weighted F1 = {lstm['f1_weighted']:.4f}.",
                f"- Final test results: baseline accuracy = {baseline['accuracy']:.4f}, LSTM accuracy = {lstm['accuracy']:.4f}.",
                "- Limitations: some files appear inconsistent with their embedded labels, the dataset is mostly tabular rather than naturally sequential, and these experiments use incremental subsets rather than the full multi-GB corpus.",
                "- Next improvement ideas: add more attack families, test a stronger tree-based baseline, evaluate scenario-aware group splitting, and compare against 1D-CNN or Transformer-style temporal models.",
                "",
            ]
        )
    (output_root / "FINAL_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end CIC-DDoS2019 LSTM project.")
    parser.add_argument("--base-dir", type=Path, default=Path.cwd(), help="Base dataset directory.")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd() / "outputs", help="Artifacts directory.")
    parser.add_argument("--phase", type=str, default="all", help="Phase name to run or 'all'.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(SEED)
    ensure_dir(args.output_dir)

    available_files = list_csv_files(args.base_dir)
    save_json(
        {"available_csv_files": [str(path.relative_to(args.base_dir)) for path in available_files]},
        args.output_dir / "dataset_inventory.json",
    )

    if args.phase == "phase3_generalization":
        run_phase3_generalization(args.base_dir, args.output_dir)
        return

    phase_configs = default_phase_configs()
    if args.phase != "all":
        phase_configs = [phase for phase in phase_configs if phase.name == args.phase]
        if not phase_configs:
            raise ValueError(f"Unknown phase: {args.phase}")

    results = []
    for phase in phase_configs:
        results.append(run_phase(args.base_dir, phase, args.output_dir))
    generate_final_report(results, args.output_dir)


if __name__ == "__main__":
    main()
