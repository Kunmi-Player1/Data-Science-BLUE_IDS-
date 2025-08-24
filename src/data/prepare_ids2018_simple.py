import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

RAW_DIR = Path("src/data/raw/ids2018/Processed Traffic Data for ML Algorithms")
OUT_DIR = Path("src/data/processed/ids2018/splits")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_csv(raw_dir : Path, rows_per_file: int = 200_000) -> pd.DataFrame:
    csvs = (raw_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"CSV not found in {raw_dir} path")
    data_frame_list = []
    for csv in csvs:
        df = pd.read_csv(csv, nrows = rows_per_file, low_memory = False, encoding_errors = "ignore")
        data_frame_list.append(df)
    return pd.concat(data_frame_list, ignore_index = True)


def normalise_columns(df : pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^A-Za-z0-9]+", "_", regex = True)
        .str.strip("_")
    )
    return df


def drop_non_numeric_cols_except_label(df: pd.DataFrame) -> pd.DataFrame:
    if 'label' not in df.columns:
        raise ValueError("Expected 'label' column in csv")
    feature_cols = [c for c in df.columns if c != 'label']
    # Try to turn everything numeric; bad parses become NaN (we’ll clean later)
    features_num = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    # Put label back
    features_num['label'] = df['label']
    return features_num


def stratified_train_val_test(x : pd.DataFrame, y : pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size = 0.7, random_state = 42, stratify = y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_test, y_test, test_size = 0.5, random_state = 42, stratify = y_test
    )
    return x_train, x_val, x_test, y_train, y_val, y_test

def coerce_and_clean_train_featureData(x_train : pd.DataFrame) -> pd.DataFrame:
    num_x_train = x_train.apply(pd.to_numeric, errors = "coerce").replace([np.inf, -np.inf], np.nan)
    all_nan_cols = num_x_train.columns[num_x_train.isna().all()].tolist()
    const_cols = num_x_train.columns[num_x_train.nunique(dropna = True) <= 1].tolist()
    cols_to_remove = list(set(all_nan_cols + const_cols))
    num_x_train.drop(columns = cols_to_remove, inplace = True)
    return num_x_train

def clean_and_align_rest_split_basedon_train_split_columns(rest_split : pd.DataFrame, train_feature_columns : list[str]) -> pd.DataFrame:
    num_rest_split = rest_split.apply(pd.to_numeric, errors = "coerce").replace([np.inf, -np.inf], np.nan)
    cols_in_trainsplit_not_in_restsplit = [col for col in train_feature_columns if col not in num_rest_split.columns]
    for c in cols_in_trainsplit_not_in_restsplit:
        num_rest_split[c] = np.nan
    return num_rest_split[train_feature_columns]

def main():
    raw_df = load_csv(RAW_DIR, rows_per_file = 200_000)
    raw_df = normalise_columns(raw_df)
    if 'timestamp' in raw_df.columns:
        raw_df.drop(columns = ['timestamp'], inplace = True)
    if 'label' not in raw_df.columns:
        raise ValueError("Expected 'label' column in csv")
    
    df_num_bool = drop_non_numeric_cols_except_label(raw_df)
    label = df_num_bool['label']
    features = df_num_bool.drop(columns = 'label')

    x_train_raw, x_val_raw, x_test_raw, y_train, y_val, y_test= stratified_train_val_test(features, label)

    x_train_clean = coerce_and_clean_train_featureData(x_train_raw)
    clean_train_feature_columns = x_train_clean.columns.tolist()

    x_val_clean = clean_and_align_rest_split_basedon_train_split_columns(x_val_raw, clean_train_feature_columns)
    x_test_clean = clean_and_align_rest_split_basedon_train_split_columns(x_test_raw, clean_train_feature_columns)

    x_train = x_train_clean.reset_index(drop = True)
    x_val = x_val_clean.reset_index(drop = True)
    x_test = x_test_clean.reset_index(drop = True)

    y_train = y_train.reset_index(drop = True)
    y_val = y_val.reset_index(drop = True)
    y_test = y_test.reset_index(drop = True)

    train_split_df = pd.concat([x_train, y_train], axis = 1)
    val_split_df = pd.concat([x_val, y_val], axis = 1)
    test_split_df = pd.concat([x_test, y_test], axis = 1)

    train_split_df.to_csv(OUT_DIR/"train.csv", index = False)
    val_split_df.to_csv(OUT_DIR/"val.csv", index = False)
    test_split_df.to_csv(OUT_DIR/"test.csv", index = False)

if __name__ == "__main__":
    main()
