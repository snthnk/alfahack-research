import numpy as np

import fa_task


def test_understands_regression_task_in_platform_layout(reg_task):
    task = fa_task.understand(reg_task)
    assert (task.task_type, task.metric, task.target, task.id_col) == ("regression", "rmse", "y", "id")
    assert task.train_path.parts[-3:] == ("regression", "participants", "train.csv")
    assert not task.greater_is_better


def test_understands_multiclass_task(clf_task):
    task = fa_task.understand(clf_task)
    assert (task.task_type, task.metric) == ("multiclass", "f1_macro")
    assert task.classes == ["hi", "low", "mid"] and task.greater_is_better
    assert task.time_col is None


def test_time_column_is_the_one_whose_test_values_all_exceed_train(reg_task):
    assert fa_task.understand(reg_task).time_col == "year"


def test_files_inside_work_dirs_are_ignored(reg_task):
    stray = reg_task.parent / "_fa_work"
    stray.mkdir()
    (stray / "train.csv").write_text("a\n1\n")
    assert fa_task.find_files(reg_task)[0].parent.name == "participants"


def test_metric_is_the_one_nearest_to_the_word_metric():
    text = "Данные сравнивались по accuracy, но основная метрика: macro F1."
    assert fa_task.detect_metric(text, "multiclass") == "f1_macro"


def test_time_holdout_is_the_latest_periods_and_never_overlaps_inner(reg_task):
    task = fa_task.understand(reg_task)
    train = fa_task.read_table(task.train_path)
    inner, hold = fa_task.split_holdout(task, train)
    assert set(inner).isdisjoint(hold) and len(inner) + len(hold) == len(train)
    assert train["year"].iloc[inner].max() < train["year"].iloc[hold].min()
    assert len(hold) >= 0.15 * len(train)


def test_random_holdout_is_stratified(clf_task):
    task = fa_task.understand(clf_task)
    train = fa_task.read_table(task.train_path)
    _, hold = fa_task.split_holdout(task, train)
    share = train["y"].value_counts(normalize=True)
    hold_share = train["y"].iloc[hold].value_counts(normalize=True)
    assert np.allclose(share.sort_index(), hold_share.sort_index(), atol=0.05)


def test_time_folds_are_expanding_windows_over_last_three_periods(reg_task):
    task = fa_task.understand(reg_task)
    train = fa_task.read_table(task.train_path)
    inner = train.iloc[fa_task.split_holdout(task, train)[0]].reset_index(drop=True)
    folds = fa_task.make_folds(task, inner)
    assert sorted(set(folds.tolist())) == [-1, 0, 1, 2]
    assert inner["year"][folds == 0].max() < inner["year"][folds == 2].min()
