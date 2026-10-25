"""Понимание задачи без LLM: файлы, таргет, тип, метрика, схема закрытой валидации.

Всё выводится из task.md и данных: sample_submission даёт id и таргет, сдвиг тестового периода относительно
train (колонка, чьи значения в test строго больше всех значений в train) — признак временной задачи.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HOLDOUT_FRAC = 0.15
MIN_INNER_FRAC = 0.5
LOWER_IS_BETTER = {"rmse", "mse", "mae", "mape", "rmsle", "log_loss"}

# (регэксп по тексту в нижнем регистре, имя метрики) — порядок важен только для равных позиций
METRIC_PATTERNS = [
    (r"macro[\s_-]*f1|f1[\s_-]*macro|макро[\s_-]*f1", "f1_macro"),
    (r"weighted[\s_-]*f1|f1[\s_-]*weighted", "f1_weighted"),
    (r"rmsle", "rmsle"),
    (r"\brmse\b|root mean squared error", "rmse"),
    (r"\bmae\b|mean absolute error", "mae"),
    (r"\bmape\b", "mape"),
    (r"\bmse\b", "mse"),
    (r"\br2\b|r\^2|r-squared", "r2"),
    (r"roc[\s_-]*auc|\bauc\b", "roc_auc"),
    (r"log[\s_-]*loss|logloss|кросс-энтропи", "log_loss"),
    (r"accuracy|точность|доля верных", "accuracy"),
    (r"\bf1\b", "f1"),
]


@dataclass
class Task:
    task_md: Path
    train_path: Path
    test_path: Path
    sample_path: Path | None
    id_col: str
    target: str
    task_type: str  # regression | binary | multiclass
    classes: list[str] = field(default_factory=list)
    metric: str = "rmse"
    time_col: str | None = None

    @property
    def greater_is_better(self) -> bool:
        return self.metric not in LOWER_IS_BETTER


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix in (".parquet", ".pq") else pd.read_csv(path)


def find_files(task_md: Path) -> tuple[Path, Path, Path | None]:
    root = task_md.parent

    def pick(stem: str) -> Path | None:
        found = [p for ext in ("csv", "parquet", "pq") for p in root.rglob(f"{stem}.{ext}")
                 if not any(part.startswith(("_", ".")) for part in p.relative_to(root).parts)]
        return min(found, key=lambda p: (len(p.relative_to(root).parts), str(p))) if found else None

    train, test = pick("train"), pick("test")
    if train is None or test is None:
        raise FileNotFoundError(f"train/test не найдены рядом с {task_md}")
    return train, test, pick("sample_submission")


def detect_metric(task_text: str, task_type: str) -> str:
    """Основная метрика: ближайшее к слову «метрик*» название метрики; нет совпадений — rmse / f1_macro."""
    text = task_text.lower()
    starts = [m.start() for m in re.finditer(r"метрик|metric|оценк", text)] or [0]
    best: tuple[int, str] | None = None
    for pattern, name in METRIC_PATTERNS:
        for m in re.finditer(pattern, text):
            dist = min(m.start() - s for s in starts if m.start() >= s) if any(m.start() >= s for s in starts) else 10**9
            if best is None or dist < best[0]:
                best = (dist, name)
    if best is not None:
        return best[1]
    return "rmse" if task_type == "regression" else "f1_macro"


def detect_task_type(y: pd.Series) -> str:
    n_unique = y.nunique(dropna=True)
    if y.dtype == object or y.dtype == bool or str(y.dtype) == "category":
        return "binary" if n_unique == 2 else "multiclass"
    if pd.api.types.is_integer_dtype(y) and n_unique <= 20:
        return "binary" if n_unique == 2 else "multiclass"
    return "regression"


def detect_time_col(train: pd.DataFrame, test: pd.DataFrame, skip: set[str]) -> str | None:
    """Колонка, у которой весь test лежит строго после train, а значений мало (год, месяц, период)."""
    candidates = []
    for col in train.columns:
        if col in skip or col not in test.columns:
            continue
        if not (pd.api.types.is_numeric_dtype(train[col]) and pd.api.types.is_numeric_dtype(test[col])):
            continue
        n = train[col].nunique()
        if 2 <= n <= 60 and test[col].notna().any() and test[col].min() > train[col].max():
            candidates.append((n, col))
    return min(candidates)[1] if candidates else None


def understand(task_md: Path) -> Task:
    train_path, test_path, sample_path = find_files(task_md)
    train, test = read_table(train_path), read_table(test_path)
    sample = read_table(sample_path) if sample_path else None

    target = id_col = None
    if sample is not None and len(sample.columns) >= 2:
        id_col = next((c for c in sample.columns if c in test.columns), sample.columns[0])
        target = next((c for c in sample.columns if c != id_col and c not in test.columns), sample.columns[-1])
    if target is None or target not in train.columns:
        extra = [c for c in train.columns if c not in test.columns]
        target = extra[0]
    if id_col is None:
        id_col = next((c for c in train.columns if c in test.columns and c.lower() in ("id", "index")), test.columns[0])

    task_type = detect_task_type(train[target])
    classes = sorted(train[target].dropna().astype(str).unique()) if task_type != "regression" else []
    metric = detect_metric(task_md.read_text(encoding="utf-8"), task_type)
    time_col = detect_time_col(train, test, {id_col, target})
    return Task(task_md, train_path, test_path, sample_path, id_col, target, task_type, classes, metric, time_col)


def split_holdout(task: Task, train: pd.DataFrame, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """(inner_idx, holdout_idx): holdout агент не видит. Временная задача — последние периоды, иначе случайные 15%."""
    n = len(train)
    if task.time_col:
        t = train[task.time_col].to_numpy()
        holdout_mask = np.zeros(n, dtype=bool)
        for value in sorted(np.unique(t[~pd.isna(t)]), reverse=True):
            if holdout_mask.sum() >= HOLDOUT_FRAC * n:
                break
            holdout_mask |= t == value
        if MIN_INNER_FRAC * n <= (~holdout_mask).sum() and holdout_mask.any():
            return np.flatnonzero(~holdout_mask), np.flatnonzero(holdout_mask)
    rng = np.random.RandomState(seed)
    stratify = train[task.target].astype(str).to_numpy() if task.task_type != "regression" else None
    holdout = np.zeros(n, dtype=bool)
    if stratify is None:
        holdout[rng.choice(n, int(HOLDOUT_FRAC * n), replace=False)] = True
    else:
        for cls in np.unique(stratify):
            idx = np.flatnonzero(stratify == cls)
            k = int(round(HOLDOUT_FRAC * len(idx))) if len(idx) >= 4 else 0
            holdout[rng.choice(idx, k, replace=False)] = True
    return np.flatnonzero(~holdout), np.flatnonzero(holdout)


def make_folds(task: Task, inner: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """Номер фолда для каждой строки inner (-1 — только для обучения). Время: расширяющееся окно по последним периодам."""
    folds = np.full(len(inner), -1)
    if task.time_col:
        t = inner[task.time_col].to_numpy()
        periods = sorted(np.unique(t[~pd.isna(t)]))
        val_periods = periods[-3:] if len(periods) >= 4 else periods[-1:]
        for i, p in enumerate(val_periods):
            folds[t == p] = i
        return folds
    from sklearn.model_selection import KFold, StratifiedKFold

    if task.task_type == "regression":
        splitter = KFold(n_splits, shuffle=True, random_state=seed).split(inner)
    else:
        y = inner[task.target].astype(str)
        splitter = StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(inner, y)
    for i, (_, va) in enumerate(splitter):
        folds[va] = i
    return folds
