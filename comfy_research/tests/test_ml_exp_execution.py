from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from comfy_research.api import ml_exp as api
from comfy_research.execution import ml_exp as service
from comfy_research.tests.train_test_fixtures import minimal_cpu_train_request

JOB = "a" * 32
SID = "action-" + "b" * 16
OPTIONS = {
    "project": "study",
    "executor": "cpu",
    "gpus": 0,
    "cpus": 2,
    "memory_gb": 4,
    "walltime_seconds": 1800,
    "datasets": {},
}


@pytest.fixture
def environment(tmp_path, monkeypatch):
    monkeypatch.setenv("COMFYRESEARCH_ML_EXP_STATE_DIR", str(tmp_path / "jobs"))
    token = tmp_path / "token"
    token.write_text("test-token")
    config = tmp_path / "client.json"
    config.write_text(
        json.dumps({"url": "https://api.example.invalid", "token_file": str(token)})
    )
    config.chmod(0o600)
    monkeypatch.setattr(
        service, "settings", lambda: {"binary": "/fake/ml-exp", "config": str(config)}
    )
    monkeypatch.setattr(service, "start", lambda job_id: None)
    monkeypatch.setattr(
        service,
        "cli",
        lambda *args, **kwargs: {
            "executors": [
                {
                    "executor_id": "cpu",
                    "enabled": True,
                    "run_creation": True,
                    "limits": {},
                }
            ]
        },
    )
    return tmp_path


def create():
    return service.create(JOB, minimal_cpu_train_request(), OPTIONS)


def test_creation_is_durable_and_duplicate_does_not_restage(environment, monkeypatch):
    first = create()
    monkeypatch.setattr(
        service, "stage_request", lambda *args: pytest.fail("duplicate restaged code")
    )
    assert create() == first
    assert service.load(service.directory(JOB))["phase"] == "PREPARED"
    assert "client" not in first
    with pytest.raises(service.JobError, match="different experiment"):
        service.create(JOB, minimal_cpu_train_request(training_steps=4), OPTIONS)


def test_pack_freezes_minimal_graph_and_independent_connection(environment):
    request = minimal_cpu_train_request()
    request["nodes"].append(
        {"id": "unrelated", "type": "comment", "data": {"secret": "unrelated-state"}}
    )
    service.create(JOB, request, OPTIONS)
    path = service.directory(JOB)
    job = service.load(path)
    assert job["client"]["config"] == str(path / "client.json")
    staged = json.loads((path / "source/request.json").read_text())
    assert all(node["id"] != "unrelated" for node in staged["nodes"])
    assert (path / "source/Dockerfile").read_text().startswith("# CPU")
    assert not (path / "source/frontend").exists()


def test_fixed_shapes_require_explicit_matching_resource_selection(
    environment, monkeypatch
):
    monkeypatch.setattr(
        service,
        "cli",
        lambda *args, **kwargs: {
            "executors": [
                {
                    "executor_id": "cpu",
                    "enabled": True,
                    "run_creation": True,
                    "limits": {
                        "fixed_resources": True,
                        "gpus": 8,
                        "cpus": 64,
                        "memory_gb": 512,
                    },
                }
            ]
        },
    )
    with pytest.raises(service.JobError, match="whole fixed shape"):
        create()
    assert not service.directory(JOB).exists()


def test_lost_submit_reply_continues_same_frozen_identity(environment, monkeypatch):
    create()
    calls = []

    def client(cfg, args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            raise service.JobError("lost reply", 503)
        return {"project": "study", "run_id": f"cr-{JOB}", "submission_id": SID}

    monkeypatch.setattr(service, "cli", client)
    service.submit(JOB)
    assert service.load(service.directory(JOB))["phase"] == "NEEDS_ATTENTION"
    service.submit(JOB)
    assert calls[0] == calls[1]
    assert "--detach" in calls[0]
    assert service.load(service.directory(JOB))["submission_id"] == SID
    service.submit(JOB)
    assert len(calls) == 2


def test_source_mutation_refuses_continue_before_any_command(environment, monkeypatch):
    create()
    (service.directory(JOB) / "source/request.json").write_text("{}")
    monkeypatch.setattr(
        service, "cli", lambda *a, **kw: pytest.fail("mutated source was submitted")
    )
    service.submit(JOB)
    assert service.load(service.directory(JOB))["phase"] == "NEEDS_ATTENTION"


def test_incomplete_source_after_crash_never_submits(environment, monkeypatch):
    create()
    (service.directory(JOB) / "source-hashes.json").unlink()
    monkeypatch.setattr(
        service, "cli", lambda *a, **kw: pytest.fail("incomplete source was submitted")
    )
    service.submit(JOB)
    assert service.load(service.directory(JOB))["phase"] == "NEEDS_ATTENTION"


def submitted():
    create()
    path = service.directory(JOB)
    job = service.load(path)
    job.update(submission_id=SID, phase="SUBMITTED")
    service.save(path, job)
    return path, job


def test_refresh_and_cancel_pin_original_submission_and_attempt(
    environment, monkeypatch
):
    path, job = submitted()
    calls = []

    def client(cfg, args, **kwargs):
        calls.append(args)
        if args[:2] == ["run", "submission"]:
            return {
                "project": "study",
                "run_id": job["run_id"],
                "submission_id": SID,
                "attempt_id": "attempt-001",
            }
        if args[0] == "status":
            assert args[-2:] == ["--attempt", "attempt-001"]
            return {
                "project": "study",
                "run_id": job["run_id"],
                "attempt_id": "attempt-001",
                "scheduler": {"state": "RUNNING"},
            }
        return {}

    monkeypatch.setattr(service, "cli", client)
    assert service.refresh(JOB)["observation"]["scheduler"]["state"] == "RUNNING"
    service.cancel(JOB)
    assert calls[-1] == ["run", "cancel", SID, "--confirmation", f"CANCEL {SID}"]
    assert service.load(path)["phase"] == "CANCEL_REQUESTED"


def test_refresh_keeps_last_evidence_on_provider_failure(environment, monkeypatch):
    path, job = submitted()
    job["observation"] = {"scheduler": {"state": "RUNNING"}}
    service.save(path, job)

    def fail(*args, **kwargs):
        raise service.JobError("offline", 503)

    monkeypatch.setattr(service, "cli", fail)
    result = service.refresh(JOB)
    assert result["observation"] == job["observation"]
    assert result["error"] == "offline"


def test_job_lock_prevents_duplicate_worker(environment, monkeypatch):
    create()
    monkeypatch.setattr(
        service, "cli", lambda *a, **kw: pytest.fail("concurrent worker submitted")
    )
    with service.locked(JOB):
        service.submit(JOB)
    assert service.load(service.directory(JOB))["phase"] == "PREPARED"


def write_result(path, envelope, *, symlink=False):
    raw = json.dumps(envelope).encode()
    with tarfile.open(path / "results.tar.gz", "w:gz") as tar:
        member = tarfile.TarInfo("comfyresearch-result.json")
        member.size = len(raw)
        if symlink:
            member.type = tarfile.SYMTYPE
            member.linkname = "/etc/passwd"
        tar.addfile(member, io.BytesIO(raw) if not symlink else None)


def test_results_require_verified_archive_and_exact_request(environment, monkeypatch):
    path, job = submitted()
    event = {"type": "complete", "loss_history": [1, 0.5], "step_ticks": [0, 1]}
    envelope = {
        "request_sha256": job["request_sha256"],
        "trainer_node_id": "trainer",
        "complete": event,
    }
    calls = []
    monkeypatch.setattr(
        service, "cli", lambda cfg, args, **kw: calls.append(args) or {}
    )
    write_result(path, envelope)
    assert service.results(JOB)["complete"] == event
    assert calls[0][:3] == ["run", "download", SID] and "--keep" in calls[0]
    write_result(path, {**envelope, "request_sha256": "0" * 64})
    with pytest.raises(service.JobError, match="different experiment"):
        service.results(JOB)
    write_result(path, envelope, symlink=True)
    with pytest.raises(service.JobError, match="bounded"):
        service.results(JOB)


def test_routes_validate_before_staging_and_get_never_submits(environment, monkeypatch):
    app = FastAPI()
    app.include_router(api.router)
    client = TestClient(app)
    invalid = minimal_cpu_train_request()
    invalid["edges"] = []
    response = client.post(
        "/api/train/ml-exp/jobs",
        json={"request_id": JOB, "train": invalid, "options": OPTIONS},
    )
    assert response.status_code == 422
    assert not service.directory(JOB).exists()
    assert (
        client.post(
            "/api/train/ml-exp/jobs",
            json={
                "request_id": JOB,
                "train": minimal_cpu_train_request(),
                "options": OPTIONS,
            },
        ).status_code
        == 202
    )
    monkeypatch.setattr(
        service, "start", lambda *args: pytest.fail("GET started a job")
    )
    assert client.get(f"/api/train/ml-exp/jobs/{JOB}").json()["phase"] == "PREPARED"
    assert client.get("/api/train/ml-exp/jobs/not-an-id").status_code == 422


def runner():
    spec = importlib.util.spec_from_file_location(
        "comfy_ml_exp_runner", service.REPO / "integrations/ml_expd/run_graph.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metric_translation_is_incremental_and_has_explicit_units():
    module = runner()
    stream = io.StringIO()
    translator = module.Translator(stream, {"obs/norm": "weight norm"})
    first = {
        "step_ticks": [0, 5],
        "loss_history": [1, 0.5],
        "observable_metric_histories": {"norm": [2, 3]},
    }
    translator.event(first)
    translator.event(first)
    translator.event({"step_ticks": [0, 5, 10], "loss_history": [1, 0.5, 0.2]})
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert len(records) == 5
    assert all(record["unit"] for record in records)
    assert records[-1] == {
        "name": "loss/train",
        "value": 0.2,
        "unit": "loss",
        "step": 10,
    }
    assert records[2]["unit"] == "weight norm"


def test_dataset_references_cannot_escape_mounted_asset(tmp_path, monkeypatch):
    module = runner()
    monkeypatch.setenv("DATASETS_JSON", json.dumps({"train": str(tmp_path)}))
    assert module.resolve_dataset_paths("ml-exp-data://train/a") == str(tmp_path / "a")
    with pytest.raises(ValueError, match="escapes"):
        module.resolve_dataset_paths("ml-exp-data://train/../private")


def test_cli_uses_argument_vector_and_does_not_leak_stderr(tmp_path):
    script = tmp_path / "fake-ml-exp"
    script.write_text("#!/bin/sh\nprintf secret-token >&2\nexit 2\n")
    script.chmod(0o700)
    with pytest.raises(service.JobError) as error:
        service.cli(
            {
                "binary": str(script),
                "config": "",
                "diagnostic": str(tmp_path / "error.log"),
            },
            ["check"],
        )
    assert "secret-token" not in str(error.value)
    assert (tmp_path / "error.log").read_text() == "secret-token"


def test_cli_retains_known_submission_when_job_already_failed(tmp_path):
    script = tmp_path / "fake-ml-exp"
    response = {
        "project": "study",
        "run_id": f"cr-{JOB}",
        "submission_id": SID,
        "job_state": "FAILED",
    }
    script.write_text("#!/bin/sh\nprintf '%s' '" + json.dumps(response) + "'\nexit 1\n")
    script.chmod(0o700)
    assert (
        service.cli({"binary": str(script), "config": ""}, ["submit"])["submission_id"]
        == SID
    )
    with pytest.raises(service.JobError):
        service.cli({"binary": str(script), "config": ""}, ["check"])


def test_batch_runner_executes_real_cpu_graph_and_preserves_complete(environment):
    pytest.importorskip("torch")
    create()
    path = service.directory(JOB)
    output = path / "outputs"
    output.mkdir()
    env = {
        **os.environ,
        "OUTPUT_DIR": str(output),
        "METRICS_PATH": str(output / "metrics.jsonl"),
        "OMP_NUM_THREADS": "1",
    }
    result = subprocess.run(
        [sys.executable, str(path / "source/run_graph.py")],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    envelope = json.loads((output / "comfyresearch-result.json").read_text())
    assert envelope["request_sha256"] == service.load(path)["request_sha256"]
    assert envelope["complete"]["type"] == "complete"
    assert envelope["complete"]["checkpoint_b64"]
    assert len(envelope["complete"]["loss_history"]) == 4
    records = [
        json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()
    ]
    assert records and all(record["unit"] for record in records)


def test_ml_exp_trainer_never_falls_back_to_local_execution():
    pytest.importorskip("torch")
    from comfy_research.api import train
    from comfy_research.schemas.train_request import TrainRequest
    from fastapi import HTTPException

    request = minimal_cpu_train_request()
    request["nodes"][-1]["data"]["executionBackend"] = "ml_exp"
    with pytest.raises(HTTPException) as error:
        train.post_train(TrainRequest.model_validate(request))
    assert error.value.status_code == 409
