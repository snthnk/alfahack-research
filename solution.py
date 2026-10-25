"""Точка входа платформы: python solution.py path/to/task.md

Свободный агент (LLM + run_python) с закрытой валидацией. Порядок:
1. задача понимается без LLM (fa_task), часть train откладывается и агенту не показывается;
2. сразу пишется submission.csv от простого LightGBM — запас на любой сбой;
3. агент улучшает pipeline.py на остальных данных, оценка — evaluate.py по заданным фолдам;
4. лучший пайплайн агента и стартовый сравниваются на отложенной части, победитель обучается на всём train.
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fa_agent  # noqa: E402
import fa_task  # noqa: E402
from fa_agent import log  # noqa: E402
from fa_metrics import evaluate_metric  # noqa: E402
from fa_templates import BASELINE_PIPELINE, EVALUATE_PY, RUNNER_PY  # noqa: E402

TOTAL_S = float(os.environ.get("FA_TOTAL_MIN", 180)) * 60  # FA_TOTAL_MIN — только для локальных коротких прогонов
HARD_EXIT_MARGIN_S = 90  # os._exit за столько до лимита: submission.csv к этому времени уже записан
FINAL_RESERVE_S = min(40 * 60, TOTAL_S / 4)  # последние минуты: проверка на отложенной части и обучение на всём train
BASELINE_TIMEOUT_S = 20 * 60


def write_submission(task: fa_task.Task, test: pd.DataFrame, labels_source: pd.Series, pred: np.ndarray, path: Path) -> None:
    if task.task_type == "regression":
        values = np.asarray(pred, dtype=float).ravel()
    else:
        original = {str(v): v for v in labels_source.dropna().unique()}
        values = np.array([original[task.classes[i]] for i in np.asarray(pred).argmax(axis=1)])
    sample_cols = list(fa_task.read_table(task.sample_path).columns) if task.sample_path else [task.id_col, task.target]
    out = pd.DataFrame({task.id_col: test[task.id_col].to_numpy(), task.target: values})[sample_cols]
    tmp = path.with_suffix(".tmp")
    out.to_csv(tmp, index=False)
    os.replace(tmp, path)


def run_pipeline(workdir: Path, pipeline: Path, train_csv: Path, pred_csv: Path, out_npy: Path, timeout: float) -> np.ndarray | None:
    """fit_predict в отдельном процессе; None — сбой или таймаут."""
    if timeout < 30:
        return None
    out_npy.unlink(missing_ok=True)
    try:
        proc = subprocess.run([sys.executable, "runner.py", str(train_csv), str(pred_csv), str(out_npy), str(pipeline)],
                              cwd=workdir, capture_output=True, text=True, timeout=timeout, start_new_session=True)
    except subprocess.TimeoutExpired:
        log(f"{pipeline.name}: таймаут {timeout:.0f} с")
        return None
    if proc.returncode != 0 or not out_npy.exists():
        log(f"{pipeline.name}: ошибка\n{proc.stderr[-1500:]}")
        return None
    pred = np.load(out_npy)
    return pred if np.isfinite(pred).all() else None


def main(task_path: str) -> None:
    t0 = time.time()
    watchdog = threading.Timer(TOTAL_S - HARD_EXIT_MARGIN_S, lambda: (log("watchdog: лимит времени, завершаем"), os._exit(0)))
    watchdog.daemon = True
    watchdog.start()
    task_md = Path(task_path).resolve()
    task = fa_task.understand(task_md)
    log(f"задача: {task.task_type}, метрика {task.metric}, таргет {task.target}, id {task.id_col}, "
        f"время-колонка {task.time_col}, классов {len(task.classes)}")
    train, test = fa_task.read_table(task.train_path), fa_task.read_table(task.test_path)
    submission = task_md.parent / "submission.csv"

    inner_idx, hold_idx = fa_task.split_holdout(task, train)
    inner, hold = train.iloc[inner_idx].reset_index(drop=True), train.iloc[hold_idx].reset_index(drop=True)
    folds = fa_task.make_folds(task, inner)
    log(f"train {len(train)} = inner {len(inner)} + holdout {len(hold)}; фолды: {sorted(set(folds.tolist()))}")

    workdir, final_dir = task_md.parent / "_fa_work", task_md.parent / "_fa_final"
    for d in (workdir, final_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir()
    inner.to_csv(workdir / "train.csv", index=False)
    test.to_csv(workdir / "test.csv", index=False)
    np.save(workdir / "folds.npy", folds)
    (workdir / "task.json").write_text(json.dumps({
        "target": task.target, "id_col": task.id_col, "task_type": task.task_type, "classes": task.classes,
        "metric": task.metric, "greater_is_better": task.greater_is_better, "time_col": task.time_col}, ensure_ascii=False))
    (workdir / "evaluate.py").write_text(EVALUATE_PY, encoding="utf-8")
    (workdir / "runner.py").write_text(RUNNER_PY, encoding="utf-8")
    (workdir / "pipeline.py").write_text(BASELINE_PIPELINE, encoding="utf-8")
    shutil.copy(HERE / "fa_metrics.py", workdir / "metrics_core.py")
    baseline = final_dir / "baseline_pipeline.py"
    baseline.write_text(BASELINE_PIPELINE, encoding="utf-8")

    # запас: стартовый пайплайн на всём train
    full_csv, test_csv = final_dir / "train_full.csv", final_dir / "test.csv"
    train.to_csv(full_csv, index=False)
    test.to_csv(test_csv, index=False)
    pred = run_pipeline(workdir, baseline, full_csv, test_csv, final_dir / "baseline_test.npy", BASELINE_TIMEOUT_S)
    if pred is None:
        pred = _constant_prediction(task, train, len(test))
        log("запасной пайплайн упал, пишем константу")
    write_submission(task, test, train[task.target], pred, submission)
    log(f"запасной submission.csv записан за {time.time() - t0:.0f} с")

    # агент
    agent_deadline = t0 + TOTAL_S - FINAL_RESERVE_S
    direction = "больше — лучше" if task.greater_is_better else "меньше — лучше"
    validation = ("Тест относится к БОЛЕЕ ПОЗДНЕМУ периоду, чем train (колонка {col}). Проверка по времени: каждый фолд — "
                  "один из последних периодов, обучение на более ранних; в fold_scores фолды идут от раннего к позднему. "
                  "Самый поздний фолд — последний в списке — ближе всего к тесту. Принимай изменение, только если оно "
                  "улучшает последний фолд, а не только среднее: прирост на ранних фолдах при ухудшении последнего — "
                  "признак подгонки под период (признаки, запоминающие год, комбинации категорий, избыточно сложные модели)."
                  ).format(col=task.time_col) if task.time_col else "Проверка: 5 фолдов."
    output_rule = (f"Для классификации верни вероятности формы (len(test), {len(task.classes)}) в порядке классов "
                   f"{task.classes} (как в task.json)." if task.classes else "Для регрессии верни вектор длины len(test).")
    system = fa_agent.SYSTEM.format(target=task.target, output_rule=output_rule, metric=task.metric, direction=direction,
                                    validation=validation, minutes=int((agent_deadline - time.time()) / 60), reserve=5)
    try:
        fa_agent.run_agent(workdir, task_md.read_text(encoding="utf-8"), system, agent_deadline - 5 * 60,
                           step_log=final_dir / "agent_messages.json")
    except Exception as e:  # агент — улучшение, а не необходимость
        log(f"агент упал: {e!r}")

    # выбор победителя на отложенной части и финальное обучение
    best = workdir / "best_pipeline.py"
    if not best.exists():
        log("best_pipeline.py нет — остаёмся на стартовом пайплайне")
        return
    hold_x = hold.drop(columns=[task.target])
    hold_csv = final_dir / "holdout_x.csv"
    hold_x.to_csv(hold_csv, index=False)
    inner_csv = workdir / "train.csv"
    left = lambda: t0 + TOTAL_S - 3 * 60 - time.time()  # noqa: E731
    scores = {}
    for name, path in (("agent", best), ("baseline", baseline)):
        p = run_pipeline(workdir, path, inner_csv, hold_csv, final_dir / f"holdout_{name}.npy", left() * 0.3)
        if p is not None and len(p) == len(hold):
            scores[name] = float(evaluate_metric(hold[task.target], p, task.metric, task.task_type, task.classes or None))
    log(f"отложенная часть ({task.metric}): {scores}")
    if "agent" not in scores:
        log("пайплайн агента не отработал на отложенной части — остаёмся на стартовом")
        return
    sign = 1 if task.greater_is_better else -1
    if "baseline" in scores and sign * (scores["agent"] - scores["baseline"]) <= 0:
        log("агент не лучше стартового пайплайна — остаёмся на нём")
        return
    pred = run_pipeline(workdir, best, full_csv, test_csv, final_dir / "agent_test.npy", left() - 60)
    if pred is None:
        log("обучение пайплайна агента на всём train не удалось — остаёмся на стартовом")
        return
    write_submission(task, test, train[task.target], pred, submission)
    log(f"submission.csv от агента записан, всего {time.time() - t0:.0f} с")


def _constant_prediction(task: fa_task.Task, train: pd.DataFrame, n: int) -> np.ndarray:
    if task.task_type == "regression":
        return np.full(n, float(train[task.target].mean()))
    freq = train[task.target].astype(str).value_counts(normalize=True)
    return np.tile([float(freq.get(c, 0.0)) for c in task.classes], (n, 1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "task.md")
