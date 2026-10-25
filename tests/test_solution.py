import numpy as np
import pandas as pd

import fa_agent
import solution


def run_with_agent(monkeypatch, task_md, agent):
    monkeypatch.setattr(fa_agent, "run_agent", agent)
    monkeypatch.setattr(solution, "TOTAL_S", 600.0)
    solution.main(str(task_md))
    return pd.read_csv(task_md.parent / "submission.csv")


def no_agent(workdir, task_text, system, deadline, **kw):
    return None


def test_baseline_submission_matches_sample_format_for_regression(monkeypatch, reg_task):
    sub = run_with_agent(monkeypatch, reg_task, no_agent)
    sample = pd.read_csv(reg_task.parent / "regression" / "participants" / "sample_submission.csv")
    test = pd.read_csv(reg_task.parent / "regression" / "participants" / "test.csv")
    assert list(sub.columns) == list(sample.columns)
    assert (sub["id"].to_numpy() == test["id"].to_numpy()).all() and sub["y"].notna().all()


def test_baseline_submission_for_multiclass_has_original_labels(monkeypatch, clf_task):
    sub = run_with_agent(monkeypatch, clf_task, no_agent)
    assert set(sub["y"]) <= {"low", "mid", "hi"} and len(sub) == 200


def test_broken_agent_pipeline_keeps_baseline(monkeypatch, reg_task):
    def broken(workdir, *a, **kw):
        (workdir / "best_pipeline.py").write_text("def fit_predict(train, test):\n    raise RuntimeError('boom')\n")

    base = run_with_agent(monkeypatch, reg_task, no_agent)
    after = run_with_agent(monkeypatch, reg_task, broken)
    assert np.allclose(base["y"], after["y"])


def test_agent_pipeline_that_beats_baseline_on_holdout_is_used(monkeypatch, reg_task):
    def good(workdir, *a, **kw):
        (workdir / "best_pipeline.py").write_text(
            "import numpy as np\n"
            "def fit_predict(train, test):\n"
            "    # линейная зависимость из генератора данных: сильнее глубокой модели на таком размере\n"
            "    return 2 * test['x1'].to_numpy() + 3 * (test['cat'] == 'a').to_numpy()\n")

    sub = run_with_agent(monkeypatch, reg_task, good)
    test = pd.read_csv(reg_task.parent / "regression" / "participants" / "test.csv")
    expected = 2 * test["x1"] + 3 * (test["cat"] == "a")
    assert np.allclose(sub["y"], expected)


def test_agent_pipeline_that_loses_on_holdout_is_rejected(monkeypatch, reg_task):
    def bad(workdir, *a, **kw):
        (workdir / "best_pipeline.py").write_text(
            "import numpy as np\ndef fit_predict(train, test):\n    return np.zeros(len(test))\n")

    base = run_with_agent(monkeypatch, reg_task, no_agent)
    after = run_with_agent(monkeypatch, reg_task, bad)
    assert np.allclose(base["y"], after["y"])
