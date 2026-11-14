"""Run the unmodified upstream CLI with a saved experiment configuration."""
import argparse
import json
import runpy
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, type=Path)
args = parser.parse_args()
config = json.loads(args.config.read_text())
project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project / "src"))
sys.argv = [str(project / "src/retrieval.py")]
for name, value in config.items():
    if isinstance(value, bool):
        if value:
            sys.argv.append(f"--{name}")
    else:
        sys.argv.extend([f"--{name}", str(value)])
runpy.run_path(str(project / "src/retrieval.py"), run_name="__main__")
