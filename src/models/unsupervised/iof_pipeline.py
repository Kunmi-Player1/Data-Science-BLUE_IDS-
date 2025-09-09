import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score

SPLITS_DIR = Path("data/processed/ids2018/splits")
SCORES_DIR = Path("data/processed/ids2018/scores")
SAVED_MODELS_DIR = Path("saved_models/iof")
SCORES_DIR.mkdir(parents=True, exist_ok=True)
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_PERCENTILES = list(np.arange(50.0, 100.0, 1.0)) + [99.5, 99.8, 99.9]

MAX_FPR = 0.345

def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(SPLITS_DIR/"train.csv", low_memory = False)
    val_df = pd.read_csv(SPLITS_DIR/"val.csv", low_memory = False)
    test_df = pd.read_csv(SPLITS_DIR/"test.csv", low_memory = False)
    if 'label' not in train_df.columns:
        raise ValueError("Expected 'label' column in splits")
    return train_df, val_df, test_df

def get_feature_columns(df : pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "label"]

def train_isolation_forest_on_benign_rows(train_df : pd.DataFrame, feature_columns : list[str]) -> IsolationForest:
    benign_rows_indicator = train_df['label'].str.strip().str.casefold().eq('benign').fillna(False)

    if benign_rows_indicator.sum() == 0:
        raise ValueError("No benign rows found in train split")
    df_benign_rows_without_label = train_df.loc[benign_rows_indicator, feature_columns]

    model = IsolationForest(
        n_estimators = 900,
        max_samples = 8192,
        max_features = 0.6,
        bootstrap = True,
        random_state = 42,
        n_jobs = -1
    )

    model.fit(df_benign_rows_without_label)

    return model

def calc_weirdness_scores(iof_model : IsolationForest, df : pd.DataFrame, feature_columns : list[str]) -> pd.Series:
    raw_scores = iof_model.decision_function(df[feature_columns])
    wierdness = -raw_scores
    return pd.Series(wierdness, index = df.index, name = "wierdness score")

def select_best_recall_precision_percentile_on_val(val_wierdness_score : pd.Series, val_df : pd.DataFrame, candidate_percentiles : list[float]) -> dict:
    is_benign = val_df['label'].str.strip().str.casefold().eq('benign').fillna(False)
    is_attack = ~is_benign

    results = []
    for p in candidate_percentiles:
        cutoff_value = float(np.percentile(val_wierdness_score[is_benign], p))
        flagged_as_attack = (val_wierdness_score >= cutoff_value)
        recall_value = recall_score(is_attack.values, flagged_as_attack.values, zero_division = 0)
        precision_value = precision_score(is_attack.values, flagged_as_attack.values, zero_division = 0)
        fpr_value = flagged_as_attack[is_benign].mean()
        f1_value = float(f1_score(is_attack.values, flagged_as_attack.values, zero_division=0))

        results.append({
            "percentile" : float(p),
            "cutoff" : float(cutoff_value),
            "recall" : float(recall_value),
            "precision" : float(precision_value),
            "fpr" : fpr_value,
            "f1" : f1_value
        })
    
    allowed = [r for r in results if r["fpr"] <= MAX_FPR]
    pool = allowed if allowed else results
    best_result = max(pool, key=lambda r: (r["recall"], r["f1"], -r["fpr"], r["precision"], r["percentile"]))
    return{"chosen" : best_result, "all_results" : results}

def save_scores_to_csv(row_index : pd.Index, wierdness_scores : pd.Series, attack_flags : pd.Series, out_path : Path):
    out_df = pd.DataFrame({
        "row_id" : row_index,
        "wierdness_score" : wierdness_scores.values,
        "iof_flag" : attack_flags.values.astype(bool)
    })

    out_df.to_csv(out_path, index = False)

def save_model_and_cutoff_info(
        iof_model : IsolationForest,
        cutoff_selection_info : dict,
        save_dir : Path = SAVED_MODELS_DIR
):
    
    model_path = save_dir / "iof_model.joblib"
    dump(iof_model, model_path)

    json_path = save_dir / "iof_threshold.json"
    json_text = json.dumps(cutoff_selection_info, indent=2, ensure_ascii=False)
    json_path.write_text(json_text, encoding = "utf-8")

def main():
    train_df, val_df, test_df = load_splits()

    feature_columns = get_feature_columns(train_df)

    benign_rows_indicator = train_df['label'].astype(str).str.strip().str.casefold().eq('benign').fillna(False)
    imputer = SimpleImputer(strategy="median")
    imputer.fit(train_df.loc[benign_rows_indicator, feature_columns])

    train_df.loc[:, feature_columns] = imputer.transform(train_df[feature_columns])
    val_df.loc[:, feature_columns]   = imputer.transform(val_df[feature_columns])
    test_df.loc[:, feature_columns]  = imputer.transform(test_df[feature_columns])

    iof_model = train_isolation_forest_on_benign_rows(train_df, feature_columns)

    val_wierdness_scores = calc_weirdness_scores(iof_model, val_df, feature_columns)
    best_cutoff_info = select_best_recall_precision_percentile_on_val(val_wierdness_scores, val_df, CANDIDATE_PERCENTILES)
    best_recall = best_cutoff_info["chosen"]["recall"]
    best_precision = best_cutoff_info["chosen"]["precision"]
    best_cutoff = best_cutoff_info["chosen"]["cutoff"]
    best_percentile = best_cutoff_info["chosen"]["percentile"]
    best_f1 = best_cutoff_info['chosen']['f1']
    
    val_attack_flags = (val_wierdness_scores >= best_cutoff)
    save_scores_to_csv(
        val_df.index,
        val_wierdness_scores,
        val_attack_flags,
        SCORES_DIR / "val_iof_scores.csv",
    )

    test_wierdness_scores = calc_weirdness_scores(iof_model, test_df, feature_columns)
    test_attack_flags = (test_wierdness_scores >= best_cutoff)

    save_scores_to_csv(
        test_df.index,
        test_wierdness_scores,
        test_attack_flags,
        SCORES_DIR / "test_iof_scores.csv",
    )

    save_model_and_cutoff_info(iof_model, best_cutoff_info, save_dir=SAVED_MODELS_DIR)

    print("IsolationForest baseline — done.")
    print(f"Best cutoff: {best_cutoff:.6f}  (percentile = {best_percentile})")
    print(f"Validation recall = {best_recall:.4f}  precision = {best_precision:.4f} F1 = {best_f1:.4f}")
    print(f"Saved: {SCORES_DIR/'val_iof_scores.csv'}, {SCORES_DIR/'test_iof_scores.csv'}, "
          f"{SAVED_MODELS_DIR/'iof_model.joblib'}, {SAVED_MODELS_DIR/'iof_threshold.json'}")
    
if __name__ == "__main__":
    main()
