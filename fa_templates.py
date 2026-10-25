"""Файлы, которые кладутся в рабочий каталог агента: стартовый pipeline.py и evaluate.py."""

BASELINE_PIPELINE = '''"""Стартовый пайплайн: LightGBM почти по умолчанию. fit_predict(train, test) -> np.ndarray."""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

TASK = json.load(open("task.json", encoding="utf-8"))


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    target = TASK["target"]
    X = train.drop(columns=[c for c in (target, TASK["id_col"]) if c in train.columns])
    Xt = test[X.columns].copy()
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) == "category":
            cats = sorted(X[c].dropna().astype(str).unique())
            X[c] = pd.Categorical(X[c].astype("string"), categories=cats)
            Xt[c] = pd.Categorical(Xt[c].astype("string"), categories=cats)
    params = dict(n_estimators=500, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, random_state=0, verbose=-1, n_jobs=8)
    if TASK["task_type"] == "regression":
        return lgb.LGBMRegressor(**params).fit(X, train[target]).predict(Xt)
    classes = TASK["classes"]
    index = {c: i for i, c in enumerate(classes)}
    y = train[target].astype(str).map(index).to_numpy()
    model = lgb.LGBMClassifier(**params).fit(X, y)
    proba = np.zeros((len(Xt), len(classes)))
    proba[:, model.classes_] = model.predict_proba(Xt)
    return proba
'''

EVALUATE_PY = '''"""Официальная оценка pipeline.py: python evaluate.py. Фолды и метрика заданы, менять нельзя."""
import importlib
import json
import sys
import time

import numpy as np
import pandas as pd

from metrics_core import evaluate_metric

task = json.load(open("task.json", encoding="utf-8"))
train = pd.read_csv("train.csv")
folds = np.load("folds.npy")
sys.path.insert(0, ".")
pipeline = importlib.import_module("pipeline")

y = train[task["target"]]
is_cls = task["task_type"] != "regression"
k = len(task["classes"])
oof = np.zeros((len(train), k)) if is_cls else np.zeros(len(train))
scores, started = [], time.time()
for f in sorted(set(folds[folds >= 0].tolist())):
    va = folds == f
    if task["time_col"]:
        tr = (train[task["time_col"]] < train.loc[va, task["time_col"]].iloc[0]).to_numpy()
    else:
        tr = folds != f
    valid = train[va].drop(columns=[task["target"]]).reset_index(drop=True)
    pred = np.asarray(pipeline.fit_predict(train[tr].reset_index(drop=True), valid), dtype=float)
    pred = pred.reshape((int(va.sum()), k) if is_cls else (int(va.sum()),))
    if not np.isfinite(pred).all():
        raise ValueError("fit_predict вернул NaN/inf")
    oof[va] = pred
    scores.append(float(evaluate_metric(y[va], pred, task["metric"], task["task_type"], task["classes"] or None)))
mask = folds >= 0
total = float(evaluate_metric(y[mask], oof[mask], task["metric"], task["task_type"], task["classes"] or None))
print(json.dumps({"score": round(total, 6), "fold_scores": [round(s, 6) for s in scores],
                  "cv_time_s": round(time.time() - started, 1)}))
'''

RUNNER_PY = '''"""fit_predict(pipeline) на файле обучения -> предсказание на файле для прогноза: runner.py train pred out.npy"""
import importlib.util
import sys

import numpy as np
import pandas as pd

train = pd.read_csv(sys.argv[1])
test = pd.read_csv(sys.argv[2])
spec = importlib.util.spec_from_file_location("pipeline_under_test", sys.argv[4])
module = importlib.util.module_from_spec(spec)
sys.path.insert(0, ".")
spec.loader.exec_module(module)
np.save(sys.argv[3], np.asarray(module.fit_predict(train, test), dtype=float))
'''
