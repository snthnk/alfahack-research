"""Сборка ZIP для платформы: python build_zip.py [-o dist/solution.zip]. Только корневые .py и requirements.txt."""
import argparse
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ["solution.py", "fa_agent.py", "fa_task.py", "fa_metrics.py", "fa_templates.py"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=str(HERE / "dist" / "solution.zip"))
    out = Path(ap.parse_args().out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in FILES:
            z.write(HERE / name, name)
        z.write(HERE / "platform" / "requirements.txt", "requirements.txt")
    print(out, out.stat().st_size, "bytes")
