import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

TRAIN_FILE = "data/composition_dataset_all.xlsx"
NEW_DATA_FILE = "data/data_4yuan.xlsx"
THRESHOLD = 0.42
OUTPUT_WITH_LABEL = "out/new_predictions_with_metrics_4yuan_composition.xlsx"
OUTPUT_NO_LABEL = "out/new_predictions_4yuan_composition.xlsx"

XGB_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 3,
    "subsample": 0.75,
    "colsample_bytree": 0.6,
    "gamma": 0.1,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "random_state": 42,
    "use_label_encoder": False,
    "verbosity": 0,
    "objective": "binary:logistic",
    "eval_metric": "auc",
}


def _ratio(num, den):
    return num / den if den else np.nan


def _metrics(tn, fp, fn, tp):
    n = tp + tn + fp + fn
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "TPR": _ratio(tp, tp + fn),
        "FNR": _ratio(fn, tp + fn),
        "TNR": _ratio(tn, tn + fp),
        "FPR": _ratio(fp, tn + fp),
        "PPV": _ratio(tp, tp + fp),
        "FDR": _ratio(fp, tp + fp),
        "NPV": _ratio(tn, tn + fn),
        "FOR": _ratio(fn, tn + fn),
        "ACC": _ratio(tp + tn, n),
    }


def load_train(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_excel(path, sheet_name=0)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].astype(int).values
    return X, y, df


def load_predict(path, ncols):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_excel(path, sheet_name=0)
    has_label = False
    y = None
    if df.shape[1] == ncols:
        last = df.iloc[:, -1].dropna().unique()
        if set(last).issubset({0, 1}):
            has_label = True
            y = df.iloc[:, -1].astype(int).values
            X = df.iloc[:, :-1].values
            return df, X, has_label, y
    return df, df.values, has_label, y


def main():
    X_train, y_train, train_df = load_train(TRAIN_FILE)
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)

    df_new, X_new, has_label, y_new = load_predict(NEW_DATA_FILE, train_df.shape[1])
    proba = model.predict_proba(X_new)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)

    out = df_new.copy()
    out["y_pred_proba"] = proba
    out[f"y_pred_thr_{THRESHOLD}"] = pred

    if has_label:
        tn, fp, fn, tp = confusion_matrix(y_new, pred, labels=[0, 1]).ravel()
        try:
            auc = roc_auc_score(y_new, proba)
        except ValueError:
            auc = np.nan
        summary = {"AUC": auc, **_metrics(tn, fp, fn, tp)}
        with pd.ExcelWriter(OUTPUT_WITH_LABEL, engine="openpyxl") as w:
            out.to_excel(w, sheet_name="predictions", index=False)
            pd.DataFrame([summary]).to_excel(w, sheet_name=f"metrics_thr_{THRESHOLD}", index=False)
        print(OUTPUT_WITH_LABEL)
    else:
        out.to_excel(OUTPUT_NO_LABEL, index=False)
        print(OUTPUT_NO_LABEL)


if __name__ == "__main__":
    main()
