from __future__ import annotations

import csv
import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from api import deps


def _install_route_test_stubs() -> None:
    fastapi = types.ModuleType("fastapi")

    class APIRouter:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def get(self, *args, **kwargs):
            def _decorator(fn):
                return fn

            return _decorator

        def post(self, *args, **kwargs):
            def _decorator(fn):
                return fn

            return _decorator

    def Query(default=None, *args, **kwargs):
        return default

    fastapi.APIRouter = APIRouter
    fastapi.Query = Query
    sys.modules["fastapi"] = fastapi

    fastapi_responses = types.ModuleType("fastapi.responses")

    class StreamingResponse:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    fastapi_responses.StreamingResponse = StreamingResponse
    sys.modules["fastapi.responses"] = fastapi_responses

    pydantic = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, **kwargs) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    pydantic.BaseModel = BaseModel
    sys.modules["pydantic"] = pydantic


_install_route_test_stubs()
annotation_routes = importlib.import_module("api.routes.annotation")
agents_routes = importlib.import_module("api.routes.agents")
data_routes = importlib.import_module("api.routes.data")
pipeline_routes = importlib.import_module("api.routes.pipeline")


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


class ApiRuntimeContractTests(unittest.TestCase):
    def test_api_config_round_trip_filters_unknown_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "pipeline_config.yaml"
            payload = {
                "api_provider": "openai",
                "api_model": "gpt-4o-mini",
                "sample_id": "DLPFC_151507",
                "visual_factor": 0.42,
                "unknown_field": "drop-me",
            }

            with patch.object(deps, "CONFIG_PATH", cfg_path):
                deps.save_config(payload)
                loaded = deps.load_config()

        self.assertEqual(loaded["api_provider"], "openai")
        self.assertEqual(loaded["api_model"], "gpt-4o-mini")
        self.assertEqual(loaded["sample_id"], "DLPFC_151507")
        self.assertEqual(float(loaded["visual_factor"]), 0.42)
        self.assertNotIn("unknown_field", loaded)

    def test_scores_route_reads_sample_specific_consensus_output(self) -> None:
        sample_id = "DLPFC_151507"
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)

            shared_dir = repo_root / "scoring" / "output" / "consensus"
            _write_csv(
                shared_dir / "scores_matrix.csv",
                ["", "methodA"],
                [["spot-1", "99"]],
            )
            _write_csv(
                shared_dir / "labels_matrix.csv",
                ["", "methodA"],
                [["spot-1", "9"]],
            )

            sample_dir = repo_root / "scoring" / "output" / sample_id / "consensus"
            _write_csv(
                sample_dir / "scores_matrix.csv",
                ["", "methodA"],
                [["spot-1", "1.5"], ["spot-2", "2.5"]],
            )
            _write_csv(
                sample_dir / "labels_matrix.csv",
                ["", "methodA"],
                [["spot-1", "1"], ["spot-2", "2"]],
            )

            with patch.object(data_routes, "_REPO_ROOT", repo_root):
                result = data_routes.get_scores(sample_id=sample_id)

        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(result["rows"][0]["method"], "methodA")
        self.assertEqual(result["rows"][0]["scores"], {"1": 1.5, "2": 2.5})

    def test_annotation_route_prefers_sample_specific_output_over_shared(self) -> None:
        sample_id = "DLPFC_151507"
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)

            shared = repo_root / "output" / "best" / "annotation_output" / "domain_annotations.json"
            shared.parent.mkdir(parents=True, exist_ok=True)
            shared.write_text(
                json.dumps(
                    [
                        {
                            "domain_id": 1,
                            "biological_identity": "Shared Label",
                            "biological_identity_conf": 0.1,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            sample_specific = (
                repo_root
                / "output"
                / "best"
                / sample_id
                / "annotation_output"
                / "domain_annotations.json"
            )
            sample_specific.parent.mkdir(parents=True, exist_ok=True)
            sample_specific.write_text(
                json.dumps(
                    [
                        {
                            "domain_id": 1,
                            "biological_identity": "Sample Label",
                            "biological_identity_conf": 0.95,
                            "function": "Sample-specific interpretation",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with patch.object(annotation_routes, "_REPO_ROOT", repo_root):
                result = annotation_routes.get_annotation(sample_id=sample_id, cluster_id=1)

        self.assertEqual(result["label"], "Sample Label")
        self.assertEqual(result["confidence"], 0.95)
        self.assertEqual(result["interpretation"], "Sample-specific interpretation")

    def test_pipeline_skip_route_accepts_legacy_stage_field(self) -> None:
        request = type("Req", (), {"name": "", "stage": "scoring"})()
        response = pipeline_routes.skip_stage(request)
        self.assertTrue(response["ok"])

        stages = {item["name"]: item for item in pipeline_routes.get_status()["stages"]}
        self.assertEqual(stages["scoring"]["status"], "skipped")
        self.assertEqual(stages["scoring"]["progress"], 100)

    def test_data_route_no_longer_exports_dead_labels_endpoint(self) -> None:
        self.assertFalse(hasattr(data_routes, "get_labels"))

    def test_agents_route_derives_status_from_pipeline_runtime(self) -> None:
        with patch.dict(
            pipeline_routes._stage_state,
            {
                "tool_runner": {"status": "running", "progress": 25},
                "scoring": {"status": "skipped", "progress": 100},
                "best": {"status": "error", "progress": 0},
                "annotation": {"status": "done", "progress": 100},
            },
            clear=True,
        ):
            response = agents_routes.get_agent_status()

        self.assertTrue(response["live"])
        self.assertIn("pipeline", response["message"].lower())
        agents = {item["id"]: item for item in response["agents"]}
        self.assertEqual(sorted(agents), ["AA", "BB", "SA", "TR"])
        self.assertEqual(agents["TR"]["status"], "ACTIVE")
        self.assertEqual(agents["TR"]["progress"], 25)
        self.assertEqual(agents["SA"]["status"], "SKIPPED")
        self.assertFalse(agents["SA"]["canSkip"])
        self.assertEqual(agents["BB"]["status"], "ERROR")
        self.assertEqual(agents["AA"]["status"], "DONE")


if __name__ == "__main__":
    unittest.main()
