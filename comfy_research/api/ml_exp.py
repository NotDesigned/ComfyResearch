"""ML-Exp batch execution. Browser requests contain experiments, never credentials."""

from __future__ import annotations

import json
import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from comfy_research.execution import ml_exp
from comfy_research.schemas.train_request import TrainRequest

router = APIRouter(prefix="/api/train/ml-exp", tags=["ml-exp"])


class JobOptions(BaseModel):
    project: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    executor: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    gpus: int = Field(default=0, ge=0, le=8)
    cpus: int = Field(default=2, ge=1, le=256)
    memory_gb: int = Field(default=4, ge=1, le=2048)
    walltime_seconds: int = Field(default=1800, ge=60, le=604800)
    datasets: dict[str, str] = Field(default_factory=dict, max_length=32)


class CreateJob(BaseModel):
    request_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    train: TrainRequest
    options: JobOptions


def call(operation, *args):
    try:
        return operation(*args)
    except ml_exp.JobError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc


@router.get("/connection")
def connection() -> dict:
    return call(ml_exp.connection)


@router.post("/jobs", status_code=202)
def create(body: CreateJob) -> dict:
    request = body.train.model_dump(mode="json")
    if len(json.dumps(request).encode()) > 8 * 1024 * 1024:
        raise HTTPException(
            413,
            "Graph exceeds the 8 MiB request limit; use data assets for large inputs",
        )
    trainers = [
        n
        for n in body.train.nodes
        if n.id == body.train.trainer_node_id and n.type == "trainer"
    ]
    if len(trainers) != 1:
        raise HTTPException(
            422, "ML-Exp execution requires one ordinary Trainer target"
        )
    ids = {node.id for node in body.train.nodes}
    if len(ids) != len(body.train.nodes):
        raise HTTPException(422, "Graph node IDs must be unique")
    for handle in ("dataset", "model", "optimizer", "loss"):
        incoming = [
            edge
            for edge in body.train.edges
            if edge.target == body.train.trainer_node_id
            and edge.targetHandle == handle
            and edge.source in ids
        ]
        if len(incoming) != 1:
            raise HTTPException(422, f"Trainer needs exactly one {handle} input")
    for name, asset in body.options.datasets.items():
        if not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name
        ) or not re.fullmatch(r"asset\.[a-f0-9]{64}", asset):
            raise HTTPException(422, "Dataset bindings must be NAME=asset.SHA256")
    return call(ml_exp.create, body.request_id, request, body.options.model_dump())


@router.get("/jobs/{job_id}")
def get(job_id: str) -> dict:
    return call(lambda: ml_exp.view(ml_exp.load(ml_exp.directory(job_id))))


@router.post("/jobs/{job_id}/continue", status_code=202)
def continue_job(job_id: str) -> dict:
    job = get(job_id)
    if job["phase"] in {"STAGING", "STAGING_FAILED"}:
        raise HTTPException(409, "Staging failed; create a new experiment")
    ml_exp.start(job_id)
    return job


@router.post("/jobs/{job_id}/refresh")
def refresh(job_id: str) -> dict:
    return call(ml_exp.refresh, job_id)


@router.post("/jobs/{job_id}/cancel")
def cancel(job_id: str) -> dict:
    return call(ml_exp.cancel, job_id)


@router.post("/jobs/{job_id}/results")
def results(job_id: str) -> dict:
    return call(ml_exp.results, job_id)
