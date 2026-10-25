"""Свободный агент: LLM + run_python. Решение — pipeline.py, оценку считает evaluate.py по заданным фолдам."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_MODEL = "qwen/qwen3.8-27b"  # платформа может не задать OPENAI_MODEL: прокси подставит модель задачи
MAX_TOOL_S = 1800
OUTPUT_CHARS = 6000
CONTEXT_CHARS = 220_000
LLM_TIMEOUT_S = 600

SYSTEM = """Ты сильный ML-инженер уровня Kaggle Grandmaster. Реши табличную задачу с помощью инструмента run_python: \
он выполняет Python в рабочем каталоге, каждый вызов — новый процесс (файлы сохраняются, переменные — нет).

Данные в рабочем каталоге: train.csv (с таргетом "{target}"), test.csv (без таргета, для справки о признаках), \
task.json, folds.npy, evaluate.py, pipeline.py. Пути к данным из описания задачи не используй — таких файлов здесь нет.

Правила:
- Решение — файл pipeline.py с функцией fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray. \
train содержит таргет, test — нет. {output_rule} Всё обучение (признаки, кодирование, веса, пороги) — только внутри \
fit_predict по переданному train. Обращаться к train.csv/test.csv/folds.npy внутри fit_predict нельзя.
- Оценка: `python evaluate.py` — обучает fit_predict на каждом фолде заданной проверки и печатает JSON со score. \
Метрика {metric}, {direction}. {validation} Фолды не меняй и свою проверку не подменяй.
- Стартовый pipeline.py (LightGBM) уже работает. Каждую версию, улучшившую score, копируй в best_pipeline.py; \
при ухудшении откатывайся. Итоговый ответ берётся из best_pipeline.py.

Время: всего {minutes} мин, остаток пишется после каждого результата. За {reserve} мин до конца прекрати \
эксперименты и убедись, что в best_pipeline.py лежит лучшая рабочая версия. После тебя пайплайн ещё раз проверят \
на отложенной части, которой у тебя нет, и один раз обучат на всех данных — учитывай время одного обучения.

Как работать: изучи данные и task.md, оцени стартовый pipeline, затем улучшай: признаки по смыслу полей, сильные \
модели (LightGBM / CatBoost / XGBoost), ансамбли, веса классов и пороги под метрику. Меняй по одной идее за шаг, \
сравнивай по score evaluate.py и не гонись за шумом: прирост меньше разброса между фолдами ничего не значит."""

TOOLS = [{
    "type": "function",
    "function": {
        "name": "run_python",
        "description": "Выполнить Python в рабочем каталоге и вернуть stdout, stderr и код возврата.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
    },
}]


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def run_python(code: str, cwd: Path, timeout: float) -> str:
    """Код в отдельном процессе; по таймауту убивается вся группа процессов."""
    proc = subprocess.Popen([sys.executable, "-u", "-c", code], cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, start_new_session=True,
                            env={"PATH": os.defpath, "PYTHONIOENCODING": "utf-8", "PYTHONPATH": str(cwd)})
    try:
        out, err = proc.communicate(timeout=max(timeout, 10))
        return json.dumps({"stdout": out[-OUTPUT_CHARS:], "stderr": err[-OUTPUT_CHARS:], "exit_code": proc.returncode},
                          ensure_ascii=False)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        return json.dumps({"error": f"превышен лимит {timeout:.0f} с, процесс убит"}, ensure_ascii=False)


def trim_context(messages: list[dict], limit: int = CONTEXT_CHARS) -> None:
    """Старые длинные сообщения обрезаются, пока контекст не влезет; system и первый user остаются целыми."""
    def size() -> int:
        return sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)

    i = 2
    while size() > limit and i < len(messages) - 6:
        m = messages[i]
        if m.get("role") == "tool" and len(m.get("content", "")) > 300:
            m["content"] = m["content"][:300] + " …[обрезано]"
        elif m.get("role") == "assistant":
            for call in m.get("tool_calls") or []:
                args = call["function"]["arguments"]
                if len(args) > 400:
                    call["function"]["arguments"] = json.dumps({"code": "…[обрезано]"})
            if len(m.get("content") or "") > 400:
                m["content"] = m["content"][:400] + " …"
        i += 1


def run_agent(workdir: Path, task_text: str, system: str, deadline: float, *, step_log: Path | None = None) -> None:
    from openai import OpenAI

    client = OpenAI()
    model = os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL
    messages = [{"role": "system", "content": system}, {"role": "user", "content": task_text}]
    errors = 0
    while time.time() + 60 < deadline:
        try:
            resp = client.chat.completions.create(model=model, messages=messages, tools=TOOLS,
                                                  parallel_tool_calls=False, timeout=LLM_TIMEOUT_S)
            errors = 0
        except Exception as e:
            errors += 1
            log(f"LLM error ({errors}): {str(e)[:200]}")
            time.sleep(min(10 * errors, 60))
            continue
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            log("агент закончил сам")
            break
        for call in msg.tool_calls:
            try:
                code = json.loads(call.function.arguments)["code"]
                out = run_python(code, workdir, min(MAX_TOOL_S, deadline - time.time()))
            except (ValueError, KeyError) as e:
                out = f"Ошибка аргументов инструмента: {e}"
            left = max(0.0, (deadline - time.time()) / 60)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": f"{out}\n[осталось {left:.0f} мин]"})
        trim_context(messages)
        log(f"шаг {len(messages)}, осталось {max(0.0, (deadline - time.time()) / 60):.0f} мин")
        if step_log:
            step_log.write_text(json.dumps(messages, ensure_ascii=False, indent=1), encoding="utf-8")
