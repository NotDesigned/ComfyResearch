"""Durable ComfyResearch jobs backed by the native ml-exp client.

Credentials and the executable are operator settings, never graph fields. Each
job freezes its source before submission. Repeating a creation reads that job;
only an explicit continue retries its original, deterministic ml-exp submission.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Iterator

from comfy_research.remote.execution_artifacts import (
    externalize_checkpoint_artifacts,
    write_artifact_blobs,
)
from comfy_research.remote.execution_graph import select_execution_graph

REPO = Path(__file__).resolve().parents[2]
ID = re.compile(r"[a-f0-9]{32}\Z")
ACTION = re.compile(r"action-[a-f0-9]{16}\Z")
MAX_RESULT = 128 * 1024 * 1024
# ML-Expd archives everything below OUTPUT_DIR under "outputs/".
RESULT_MEMBER = "outputs/comfyresearch-result.json"
# A background submission waits this long for a short refresh to release the job.
SUBMIT_LOCK_WAIT = 10.0


class JobError(Exception):
    def __init__(self, message: str, status: int = 409):
        super().__init__(message)
        self.status = status


class JobBusy(JobError):
    """Another request holds this job's lock."""


def root() -> Path:
    path = Path(
        os.environ.get("COMFYRESEARCH_ML_EXP_STATE_DIR", ".comfyresearch/ml-exp")
    ).resolve()
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def directory(job_id: str) -> Path:
    if not ID.fullmatch(job_id):
        raise JobError("Invalid ML-Exp job ID", 422)
    return root() / job_id


@contextlib.contextmanager
def locked(job_id: str, wait: float = 0.0) -> Iterator[Path]:
    path = directory(job_id)
    deadline = time.monotonic() + wait
    with open(path.with_suffix(".lock"), "a+b") as handle:
        while True:
            try:
                if os.name == "nt":
                    import msvcrt

                    handle.write(b"0")
                    handle.flush()
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise JobBusy("This job is busy; query it again") from exc
                time.sleep(0.2)
        try:
            yield path
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def save(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path, delete=False
    ) as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
        name = stream.name
    os.replace(name, path / "job.json")


def load(path: Path) -> dict:
    try:
        return json.loads((path / "job.json").read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise JobError("ML-Exp job not found", 404) from exc


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def settings() -> dict:
    binary = shutil.which(os.environ.get("COMFYRESEARCH_ML_EXP_BINARY", "ml-exp"))
    if not binary:
        raise JobError(
            "Install ml-exp on the ComfyResearch server and configure its connection",
            503,
        )
    default = (
        Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
        / "ml-exp/client.json"
    )
    config = (
        Path(
            os.environ.get(
                "COMFYRESEARCH_ML_EXP_CONFIG",
                os.environ.get("ML_EXPD_CLIENT_CONFIG", str(default)),
            )
        )
        .expanduser()
        .resolve()
    )
    if not config.is_file():
        raise JobError(
            "Configure ml-exp with a private client.json and token file on the ComfyResearch server",
            503,
        )
    return {"binary": str(Path(binary).resolve()), "config": str(config)}


def freeze_connection(path: Path, cfg: dict) -> dict:
    original = Path(cfg["config"])
    info = original.stat()
    if os.name != "nt" and (info.st_mode & 0o077 or info.st_uid != os.getuid()):
        raise JobError(
            "ML-Exp client configuration must be private and owned by this server user",
            503,
        )
    raw = original.read_bytes()
    if len(raw) > 65536:
        raise JobError("ML-Exp client configuration is too large", 503)
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise JobError("ML-Exp client configuration is invalid JSON", 503) from exc
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("token_file"), str)
        or not value.get("url")
        or not Path(value["token_file"]).is_absolute()
    ):
        raise JobError(
            "ML-Exp configuration needs a saved URL and an absolute token_file", 503
        )
    # Freeze the server URL, not the secret itself. Rotating the referenced token remains possible.
    target = path / "client.json"
    target.write_bytes(raw)
    target.chmod(0o600)
    return {**cfg, "config": str(target), "diagnostic": str(path / "cli-error.log")}


def retain_cli_error(cfg: dict, stream) -> None:
    if cfg.get("diagnostic"):
        stream.seek(0)
        path = Path(cfg["diagnostic"])
        with open(path, "wb") as destination:
            path.chmod(0o600)
            destination.write(stream.read(8192))


def cli(cfg: dict, args: list[str], *, timeout: int = 60) -> dict:
    command = [cfg["binary"], "--json"]
    if cfg["config"]:
        command += ["--config", cfg["config"]]
    # Do not expose CLI stderr (which can contain private operator paths) to the browser.
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        try:
            env = {
                key: value
                for key, value in os.environ.items()
                if not key.startswith("ML_EXPD_")
            }
            result = subprocess.run(
                command + args,
                stdout=out,
                stderr=err,
                timeout=timeout,
                check=False,
                env=env,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            retain_cli_error(cfg, err)
            raise JobError(
                "ML-Exp command did not finish; retain this job and refresh or continue it",
                503,
            ) from exc
        if result.returncode:
            retain_cli_error(cfg, err)
        out.seek(0)
        raw = out.read(8 * 1024 * 1024 + 1)
    failure = JobError(
        f"ML-Exp {args[0]} failed (exit {result.returncode}); check the server connection and job configuration",
        502,
    )
    try:
        value = json.loads(raw)
        if len(raw) > 8 * 1024 * 1024 or not isinstance(value, dict):
            raise ValueError()
    except ValueError as exc:
        if result.returncode:
            raise failure from exc
        raise JobError("ML-Exp returned an invalid JSON response", 502) from exc
    if result.returncode and not (
        args[0] == "submit"
        and isinstance(value.get("submission_id"), str)
        and ACTION.fullmatch(value["submission_id"])
    ):
        raise failure
    # submit may print a complete, owned identity and exit nonzero because the
    # job already ended unsuccessfully. Keep it observable; never invent a new job.
    return value


def connection() -> dict:
    cfg = settings()
    health = cli(cfg, ["check"])
    backends = cli(cfg, ["backend"])
    return {
        "connected": True,
        "server_version": health.get("server_version"),
        "executors": backends.get("executors", []),
    }


def validate_executor(cfg: dict, options: dict) -> None:
    catalog = cli(cfg, ["backend"])
    row = next(
        (
            item
            for item in catalog.get("executors", [])
            if item.get("executor_id") == options["executor"]
        ),
        None,
    )
    if not row or row.get("enabled") is not True or row.get("run_creation") is not True:
        raise JobError(
            "Choose an exact, enabled executor ID from Check connection", 422
        )
    limits = row.get("limits", {})
    if limits.get("fixed_resources") is True:
        expected = {key: limits.get(key) for key in ("gpus", "cpus", "memory_gb")}
        if any(options[key] != value for key, value in expected.items()):
            raise JobError(
                f"This executor allocates its whole fixed shape: {expected['gpus']} GPUs, {expected['cpus']} CPUs, {expected['memory_gb']} GB. Select those exact resources before submitting.",
                422,
            )


def stage_request(path: Path, request: dict, options: dict) -> str:
    """Reuse the batch packer; source and environment remain separate identities."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "comfy_ml_exp_pack", REPO / "integrations/ml_expd/pack.py"
    )
    assert spec and spec.loader
    pack = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pack)
    stage = path / "source"
    stage.mkdir()
    pack.copy_package(stage)
    for name in ("requirements.txt",):
        shutil.copy2(REPO / name, stage / name)
    here = REPO / "integrations/ml_expd"
    recipe = "Dockerfile.cuda" if options["gpus"] else "Dockerfile.cpu"
    shutil.copy2(here / recipe, stage / "Dockerfile")
    shutil.copy2(here / "run_graph.py", stage / "run_graph.py")
    # Freeze local user definitions as well as built-in definitions, never credentials or datasets.
    data = stage / "data"
    data.mkdir()
    for name in (
        "user_observables.json",
        "user_linear_datasets.json",
        "user_symbolic_func_datasets.json",
    ):
        source = REPO / "data" / name
        if source.is_file():
            shutil.copy2(source, data / name)
    bundled = REPO / "data/bundled"
    if bundled.is_dir():
        shutil.copytree(bundled, data / "bundled")
    selected = select_execution_graph(request, request["trainer_node_id"])
    for node in selected["nodes"]:
        if node["id"] == request["trainer_node_id"]:
            node["data"] = {
                **node.get("data", {}),
                "computeDevice": "cuda" if options["gpus"] else "cpu",
                "remoteGpu": False,
            }
            for key in (
                "executionBackend",
                "mlExpSettings",
                "mlExpJobId",
                "mlExpAppliedJobId",
            ):
                node["data"].pop(key, None)
    selected, blobs = externalize_checkpoint_artifacts(selected)
    write_artifact_blobs(stage / "input-artifacts", blobs)
    (stage / "request.json").write_text(
        json.dumps(selected, ensure_ascii=False), encoding="utf-8"
    )
    manifest = {
        str(p.relative_to(stage)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(stage.rglob("*"))
        if p.is_file()
    }
    (path / "source-hashes.json").write_text(json.dumps(manifest), encoding="utf-8")
    return hashlib.sha256((stage / "request.json").read_bytes()).hexdigest()


def view(job: dict) -> dict:
    keys = (
        "job_id",
        "trainer_node_id",
        "phase",
        "project",
        "run_id",
        "submission_id",
        "attempt_id",
        "options",
        "error",
        "observation",
        "request_sha256",
    )
    return {key: job.get(key) for key in keys}


def create(job_id: str, request: dict, options: dict) -> dict:
    identity = digest({"request": request, "options": options})
    with locked(job_id) as path:
        if (path / "job.json").exists():
            old = load(path)
            if old["identity"] != identity:
                raise JobError(
                    "This request ID already belongs to a different experiment"
                )
            return view(old)
        cfg = settings()
        validate_executor(cfg, options)
        path.mkdir(mode=0o700, exist_ok=True)
        cfg = freeze_connection(path, cfg)
        job = {
            "job_id": job_id,
            "identity": identity,
            "trainer_node_id": request["trainer_node_id"],
            "project": options["project"],
            "run_id": f"cr-{job_id}",
            "options": options,
            "client": cfg,
            "phase": "STAGING",
            "error": None,
        }
        save(path, job)
        try:
            job["request_sha256"] = stage_request(path, request, options)
            job["phase"] = "PREPARED"
        except Exception:
            job["phase"] = "STAGING_FAILED"
            job["error"] = (
                "Could not freeze experiment files; create a new job after fixing the graph"
            )
            save(path, job)
            raise
        save(path, job)
    start(job_id)
    return view(job)


def verify_source(path: Path) -> None:
    expected = json.loads((path / "source-hashes.json").read_text(encoding="utf-8"))
    actual = {
        str(p.relative_to(path / "source")): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((path / "source").rglob("*"))
        if p.is_file()
    }
    if actual != expected:
        raise JobError("Frozen source changed; this job cannot be submitted")


@contextlib.contextmanager
def exported_source(path: Path) -> Iterator[Path]:
    """Export the sealed input outside ancestor Git ignore rules.

    Native ml-exp deliberately respects an ancestor repository's .gitignore.
    Our private state normally lives in the ignored .comfyresearch directory,
    so it cannot itself be used as the native upload root. Keep the basename
    'source': native submission identity binds that name and content, not the
    absolute temporary path. An existing failed task can therefore continue.
    """
    with tempfile.TemporaryDirectory(prefix="comfyresearch-ml-exp-") as temporary:
        source = Path(temporary).resolve() / "source"
        if any(
            (parent / ".git/HEAD").exists() or (parent / ".git").is_file()
            for parent in source.parents
        ):
            raise JobError(
                "The temporary directory is inside a Git repository. Set TMPDIR outside repositories before starting ComfyResearch."
            )
        shutil.copytree(path / "source", source)
        expected = json.loads((path / "source-hashes.json").read_text(encoding="utf-8"))
        actual = {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source.rglob("*"))
            if p.is_file()
        }
        if actual != expected:
            raise JobError(
                "Frozen source changed while exporting; nothing was submitted"
            )
        yield source


def submit(job_id: str) -> None:
    try:
        # Wait out a concurrent refresh, which holds the lock briefly; a job that
        # stays busy belongs to another submission worker.
        with locked(job_id, wait=SUBMIT_LOCK_WAIT) as path:
            job = load(path)
            if job.get("submission_id") or job["phase"] in {
                "STAGING",
                "STAGING_FAILED",
            }:
                return
            try:
                verify_source(path)
                job.update(phase="SUBMITTING", error=None)
                save(path, job)
                opt = job["options"]
                args = [
                    "submit",
                    str(path / "source"),
                    "--project",
                    job["project"],
                    "--executor",
                    opt["executor"],
                    "--run",
                    job["run_id"],
                    "--gpus",
                    str(opt["gpus"]),
                    "--cpus",
                    str(opt["cpus"]),
                    "--memory",
                    str(opt["memory_gb"]),
                    "--time",
                    str(opt["walltime_seconds"]) + "s",
                    "--yes",
                    "--detach",
                ]
                for name, asset in sorted(opt.get("datasets", {}).items()):
                    args += ["--data", f"{name}={asset}"]
                args += ["--", "python3", "-u", "run_graph.py"]
                with exported_source(path) as source:
                    args[1] = str(source)
                    result = cli(job["client"], args, timeout=3600)
                if (
                    result.get("project") != job["project"]
                    or result.get("run_id") != job["run_id"]
                    or not ACTION.fullmatch(result.get("submission_id", ""))
                ):
                    raise JobError(
                        "Submission response differs from the frozen experiment", 502
                    )
                job.update(
                    phase="SUBMITTED", submission_id=result["submission_id"], error=None
                )
            except JobError as exc:
                job.update(phase="NEEDS_ATTENTION", error=str(exc))
            except (OSError, ValueError, KeyError, TypeError):
                job.update(
                    phase="NEEDS_ATTENTION",
                    error="Could not read the frozen source or submission response; retain this task's identity",
                )
            save(path, job)
    except JobBusy:
        # Another worker owns the same job. Never submit a different identity.
        return


def start(job_id: str) -> None:
    threading.Thread(
        target=submit, args=(job_id,), daemon=True, name=f"ml-exp-{job_id[:8]}"
    ).start()


def refresh(job_id: str) -> dict:
    with contextlib.ExitStack() as stack:
        try:
            path = stack.enter_context(locked(job_id))
        except JobBusy:
            # A submission can hold the job for up to an hour while its image
            # builds. Report the saved phase rather than an error.
            return view(load(directory(job_id)))
        job = load(path)
        if not job.get("submission_id"):
            return view(job)
        cfg, sid = job["client"], job["submission_id"]
        try:
            cli(cfg, ["run", "refresh", sid])
            original = cli(cfg, ["run", "submission", sid])
            attempt = original.get("attempt_id")
            if (
                original.get("submission_id") != sid
                or original.get("project") != job["project"]
                or original.get("run_id") != job["run_id"]
                or not re.fullmatch(r"attempt-\d{3,}", attempt or "")
            ):
                raise JobError(
                    "Original submission identity could not be verified", 502
                )
            if job.get("attempt_id", attempt) != attempt:
                raise JobError("Submission Attempt changed", 502)
            observation = cli(
                cfg,
                [
                    "status",
                    "--project",
                    job["project"],
                    "--run",
                    job["run_id"],
                    "--attempt",
                    attempt,
                ],
            )
            if any(
                observation.get(k) != v
                for k, v in {
                    "project": job["project"],
                    "run_id": job["run_id"],
                    "attempt_id": attempt,
                }.items()
            ):
                raise JobError("Status differs from the frozen experiment", 502)
            job.update(attempt_id=attempt, observation=observation, error=None)
        except JobError as exc:
            job["error"] = str(exc)
        save(path, job)
        return view(job)


def cancel(job_id: str) -> dict:
    with locked(job_id) as path:
        job = load(path)
        sid = job.get("submission_id")
        if not sid:
            raise JobError(
                "Submission is not yet known; continue or refresh this job before cancelling"
            )
        cli(job["client"], ["run", "cancel", sid, "--confirmation", f"CANCEL {sid}"])
        job["phase"] = "CANCEL_REQUESTED"
        save(path, job)
        return view(job)


def results(job_id: str) -> dict:
    with locked(job_id) as path:
        job = load(path)
        if not job.get("submission_id"):
            raise JobError("No submitted job to download")
        archive = path / "results.tar"
        cli(
            job["client"],
            [
                "run",
                "download",
                job["submission_id"],
                "--output",
                str(archive),
                "--keep",
            ],
            timeout=600,
        )
        # ml-exp verifies the entire archive. Read one bounded regular file without extracting it.
        with tarfile.open(archive, "r:*") as tar:
            members = [
                m
                for m in tar
                if m.name.removeprefix("./") == RESULT_MEMBER
            ]
            if (
                len(members) != 1
                or not members[0].isfile()
                or members[0].size > MAX_RESULT
            ):
                raise JobError(
                    "Verified archive has no bounded ComfyResearch result", 422
                )
            stream = tar.extractfile(members[0])
            assert stream is not None
            envelope = json.load(stream)
        if (
            envelope.get("request_sha256") != job["request_sha256"]
            or envelope.get("trainer_node_id") != job["trainer_node_id"]
        ):
            raise JobError("Result belongs to a different experiment", 422)
        event = envelope.get("complete", {})
        if event.get("type") != "complete":
            raise JobError("The experiment has no successful complete event", 422)
        return {"job": view(job), "complete": event}
