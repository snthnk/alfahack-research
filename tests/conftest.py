import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TASK_MD = "Предскажи {target}. Метрика качества: {metric}."


def make_task_dir(root: Path, kind: str, n: int = 400, seed: int = 0, timed: bool = False) -> Path:
    """Папка в раскладке платформы: <kind>/participants/{train,test,sample_submission}.csv и task.md рядом с <kind>/."""
    rng = np.random.RandomState(seed)
    part = root / kind / "participants"
    part.mkdir(parents=True)

    def frame(m: int, years: list[int]) -> pd.DataFrame:
        df = pd.DataFrame({"id": np.arange(m), "x1": rng.randn(m), "x2": rng.randn(m), "cat": rng.choice(list("abc"), m)})
        if timed:
            df["year"] = rng.choice(years, m)
        return df

    train, test = frame(n, [1, 2, 3, 4, 5]), frame(n // 2, [6])
    signal = train["x1"] * 2 + (train["cat"] == "a") * 3
    if kind == "regression":
        train["y"] = signal + rng.randn(n) * 0.1
        metric, sample = "RMSE", pd.DataFrame({"id": test["id"], "y": 0.0})
    else:
        train["y"] = np.where(signal > 1.5, "hi", np.where(signal > 0, "mid", "low"))
        metric, sample = "macro F1", pd.DataFrame({"id": test["id"], "y": "low"})
    train.to_csv(part / "train.csv", index=False)
    test.to_csv(part / "test.csv", index=False)
    sample.to_csv(part / "sample_submission.csv", index=False)
    (root / "task.md").write_text(TASK_MD.format(target="y", metric=metric), encoding="utf-8")
    return root / "task.md"


@pytest.fixture
def reg_task(tmp_path):
    return make_task_dir(tmp_path, "regression", timed=True)


@pytest.fixture
def clf_task(tmp_path):
    return make_task_dir(tmp_path, "multiclass")
