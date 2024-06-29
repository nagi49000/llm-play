from transformers import AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from datetime import datetime
import logging
import os
import yaml
from pathlib import Path
from pydantic import BaseModel
from typing import Optional


class HealthcheckResponse(BaseModel):
    status: str


class VersionResponse(BaseModel):
    app_name: str
    githash: str
    build_time: datetime


class PipelineSpecResponse(BaseModel):
    pipeline: dict


class InvokeResponse(BaseModel):
    response: str


class InvokeRequest(BaseModel):
    query: str


def get_gunicorn_logger(name: str = "gunicorn.error"):
    logger = logging.getLogger(name)  # dirty way of wiring into the gunicorn logger
    # overwrite the root logger with the gunicorn logger
    root_logger = logging.getLogger()
    root_logger.handlers = logger.handlers
    root_logger.setLevel(logger.level)
    return logger


def get_llm_pipeline_from_yaml(filename: Optional[str] = None) -> HuggingFacePipeline:
    if filename is None:
        filename = Path(__file__).parent / "llm-pipeline.yaml"
    with open(filename) as f:
        pipeline_spec = yaml.safe_load(f)
    p = pipeline_spec["pipeline"]
    llm_pipeline = pipeline(
        p["pipeline-type"],
        model=p["model"]["model-name"],
        tokenizer=AutoTokenizer.from_pretrained(
            p["tokenizer"]["model-name"],
            **p["tokenizer"]["kwargs"]
        ),
        device_map="cpu",  # suppresses GPU use
        model_kwargs=p["model"]["kwargs"],
        **p["kwargs"]
    )

    llm = HuggingFacePipeline(
        pipeline=llm_pipeline,
    )

    return llm


def create_app(logger_name: str = "gunicorn.error", pipeline_yaml_filename: Optional[str] = None):
    logger = get_gunicorn_logger(name=logger_name)
    llm = get_llm_pipeline_from_yaml(pipeline_yaml_filename)
    app_name = os.getenv("APP_NAME", "self contained CPU model")
    # allow a root path to be specified in case the app is behind a reverse proxy or endpoint - needed for anything UI based
    root_path = os.getenv("ROOT_PATH", None)
    app = FastAPI(title=app_name, root_path=root_path)
    # specify static files for docs_url and redoc_url so that the UI docs will work offline
    static_files_dir = Path(__file__).parents[2] / "static"
    app.mount("/static", StaticFiles(directory=str(static_files_dir)), name="static")

    @app.get("/healthcheck", response_model=HealthcheckResponse)
    async def healthcheck():
        return {"status": "OK"}

    @app.get("/version", response_model=VersionResponse)
    async def version():
        logger.debug("/version")
        return {
            "app_name": app_name,
            "githash": os.getenv("GITHASH"),
            "build_time": os.getenv("BUILD_TIME")
        }

    @app.get("/pipeline-spec", response_model=PipelineSpecResponse)
    async def pipeline_spec():
        spec_filename = Path(__file__).parent / "llm-pipeline.yaml"
        with open(spec_filename) as f:
            pipeline_spec = yaml.safe_load(f)
        return pipeline_spec

    @app.post("/invoke", response_model=InvokeResponse)
    async def invoke(p: InvokeRequest):
        return {"response": llm.invoke(p.query)}

    return app
