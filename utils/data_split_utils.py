# external imports

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


COHORT_COLUMN_CANDIDATES = ("cohort_num", "cohert_num")


def _parse_cohort_spec(spec):
    """Parse a cohort spec like '1-8', '1,2,3', or '1 2 3' into sorted ints."""
    if spec is None:
        return []

    if isinstance(spec, (list, tuple, set, np.ndarray)):
        values = []
        for item in spec:
            values.extend(_parse_cohort_spec(item))
        return sorted(set(values))

    spec_text = str(spec).replace(",", " ").strip()
    if not spec_text:
        return []

    cohorts = set()
    for token in spec_text.split():
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise ValueError(f"Invalid cohort range '{token}': start must be <= end.")
            cohorts.update(range(start, end + 1))
        else:
            cohorts.add(int(token))

    return sorted(cohorts)


def get_cohort_column(df):
    """Return the cohort column name, supporting both cohort_num and cohert_num."""
    for column in COHORT_COLUMN_CANDIDATES:
        if column in df.columns:
            return column

    raise ValueError(
        "Could not find a cohort column in the input CSV. "
        f"Expected one of: {COHORT_COLUMN_CANDIDATES}"
    )


def ensure_cohort_num_column(df):
    """
    Return a copy with a normalized integer cohort_num column.

    If the input only has the legacy misspelled cohert_num column, it is copied
    into cohort_num so downstream code can rely on a single canonical name.
    """
    cohort_column = get_cohort_column(df)
    normalized_df = df.copy()
    normalized_df["cohort_num"] = pd.to_numeric(
        normalized_df[cohort_column], errors="raise"
    ).astype(int)
    return normalized_df


def split_df_by_cohorts(df, train_cohorts="1-8", test_cohorts="9-10"):
    """
    Split a dataframe into train and test subsets using cohort membership.

    Returns:
        normalized_df, train_df, test_df
    """
    normalized_df = ensure_cohort_num_column(df)

    train_cohort_values = _parse_cohort_spec(train_cohorts)
    test_cohort_values = _parse_cohort_spec(test_cohorts)

    if not train_cohort_values:
        raise ValueError("train_cohorts must contain at least one cohort.")
    if not test_cohort_values:
        raise ValueError("test_cohorts must contain at least one cohort.")

    overlap = set(train_cohort_values) & set(test_cohort_values)
    if overlap:
        raise ValueError(
            "train_cohorts and test_cohorts overlap. "
            f"Overlapping cohorts: {sorted(overlap)}"
        )

    train_df = normalized_df[normalized_df["cohort_num"].isin(train_cohort_values)].reset_index(drop=True)
    test_df = normalized_df[normalized_df["cohort_num"].isin(test_cohort_values)].reset_index(drop=True)

    if len(train_df) == 0:
        raise ValueError(
            f"No rows found for train_cohorts={train_cohort_values}. Check the cohort column values."
        )
    if len(test_df) == 0:
        raise ValueError(
            f"No rows found for test_cohorts={test_cohort_values}. Check the cohort column values."
        )

    return normalized_df, train_df, test_df


def _validate_grouped_split(train_df, val_df, label_name, context):
    """Validate there is no patient leakage and both splits have positive and negative samples."""
    train_patients = set(train_df["patient_id"].astype(str).tolist())
    val_patients = set(val_df["patient_id"].astype(str).tolist())
    overlapping_patients = train_patients & val_patients
    if overlapping_patients:
        sample_patients = sorted(overlapping_patients)[:10]
        raise ValueError(
            f"{context}: patient leakage detected between train and validation. "
            f"Example patient_ids: {sample_patients}"
        )

    train_label_values = pd.Series(train_df[label_name]).astype(int)
    val_label_values = pd.Series(val_df[label_name]).astype(int)
    if train_label_values.nunique() < 2:
        raise ValueError(
            f"{context}: training split must contain both positive and negative samples."
        )
    if val_label_values.nunique() < 2:
        raise ValueError(
            f"{context}: validation split must contain both positive and negative samples."
        )


def stratified_train_val_split(dev_df, frac_split=0.2, shuffle=True, args=None):
    if args is None:
        raise ValueError("args is required so stratified splitting can access label and seed.")
    if frac_split <= 0 or frac_split >= 1:
        raise ValueError(f"frac_split must be in (0, 1). Received: {frac_split}")

    working_df = dev_df.reset_index(drop=True)
    if shuffle:
        working_df = working_df.sample(
            frac=1, random_state=getattr(args, "seed", 0)
        ).reset_index(drop=True)

    group_patient_id_list = np.array(working_df["patient_id"].values)
    n_splits = int(round(1 / frac_split))
    if n_splits < 2:
        raise ValueError(
            f"frac_split={frac_split} results in n_splits={n_splits}, but at least 2 splits are required."
        )

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=getattr(args, "seed", 0) if shuffle else None,
    )

    train_idxs, val_idxs = next(
        sgkf.split(working_df, working_df[args.label].values, groups=group_patient_id_list)
    )

    train_df = working_df.iloc[train_idxs].reset_index(drop=True)
    val_df = working_df.iloc[val_idxs].reset_index(drop=True)
    _validate_grouped_split(train_df, val_df, args.label, "Internal train/val split")

    return train_df, val_df


def generator_cross_val_folds(dev_df, k_folds=5, label_name="", shuffle=True, random_state=0):
    if k_folds <= 1:
        raise ValueError(
            f"k_folds must be greater than 1 for cross-validation. Received: {k_folds}"
        )

    working_df = dev_df.reset_index(drop=True)
    if shuffle:
        working_df = working_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    sgkf_cross_val = StratifiedGroupKFold(
        n_splits=k_folds,
        shuffle=shuffle,
        random_state=random_state if shuffle else None,
    )

    group_patient_id_list = np.array(working_df["patient_id"].values)

    for fold_idx, (train_idxs, val_idxs) in enumerate(
        sgkf_cross_val.split(
            working_df,
            working_df[label_name].values,
            groups=group_patient_id_list,
        )
    ):
        train_fold_df = working_df.iloc[train_idxs].reset_index(drop=True)
        val_fold_df = working_df.iloc[val_idxs].reset_index(drop=True)

        _validate_grouped_split(
            train_fold_df,
            val_fold_df,
            label_name,
            f"Cross-validation fold {fold_idx}",
        )

        yield train_fold_df, val_fold_df
