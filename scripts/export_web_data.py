"""Export saved local experiment summaries for a static GitHub Pages dashboard."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from web_demo import experiment_data  # noqa: E402


def main() -> None:
    output = ROOT / "docs" / "data.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(experiment_data(), indent=2), encoding="utf-8")
    print(f"Exported dashboard data to {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

