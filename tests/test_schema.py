from __future__ import annotations

import json
from pathlib import Path

from sre_bench.schema import Scenario


def test_all_scenarios_validate() -> None:
    scenario_dir = Path("scenarios")
    scenario_paths = sorted(scenario_dir.glob("*.json"))

    assert len(scenario_paths) == 8

    ids: list[str] = []
    for path in scenario_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        scenario = Scenario.model_validate(payload)
        ids.append(scenario.id)

    assert len(ids) == len(set(ids))
