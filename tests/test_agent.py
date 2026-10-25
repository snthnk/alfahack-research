import json
import time

import fa_agent


def test_run_python_returns_stdout_and_exit_code(tmp_path):
    out = json.loads(fa_agent.run_python("print(2 + 2)", tmp_path, 30))
    assert out["stdout"].strip() == "4" and out["exit_code"] == 0


def test_run_python_kills_the_process_on_timeout(tmp_path):
    started = time.time()
    out = json.loads(fa_agent.run_python("import time; time.sleep(60)", tmp_path, 10))
    assert "error" in out and time.time() - started < 30


def test_trim_context_shortens_old_tool_outputs_but_keeps_recent_ones():
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    for i in range(20):
        messages.append({"role": "tool", "tool_call_id": str(i), "content": "x" * 5000})
    fa_agent.trim_context(messages, limit=30_000)
    assert len(messages[2]["content"]) < 400
    assert len(messages[-1]["content"]) == 5000
    assert messages[0]["content"] == "s"
