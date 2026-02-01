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
    "eval_metric": "auc"
}