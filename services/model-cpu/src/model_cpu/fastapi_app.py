from transformers import AutoTokenizer, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from fastapi import FastAPI
from datetime import datetime
import logging
import os
from pydantic import BaseModel


class HealthcheckResponse(BaseModel):
    status: str


class VersionResponse(BaseModel):
    app_name: str
    githash: str
    build_time: datetime


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


def get_text_generator_llm(
    model_name: str,
    model_kwargs: dict,
    tokenizer_name: str,
    tokenizer_kwargs: dict,
    max_new_tokens: int
) -> HuggingFacePipeline:
    text_generator = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=AutoTokenizer.from_pretrained(
            tokenizer_name,
            **tokenizer_kwargs
        ),
        device_map="cpu",  # suppresses GPU use
        model_kwargs=model_kwargs,
        max_new_tokens=max_new_tokens,
    )

    llm = HuggingFacePipeline(
        pipeline=text_generator,
    )

    return llm


def create_app(llm: HuggingFacePipeline, logger_name: str = "gunicorn.error"):
    logger = get_gunicorn_logger(name=logger_name)
    app_name = os.getenv("APP_NAME", "self contained CPU model")
    root_path = os.getenv("ROOT_PATH", None)
    app = FastAPI(title=app_name, root_path=root_path)

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

    @app.post("/invoke", response_model=InvokeResponse)
    async def invoke(p: InvokeRequest):
        return {"response": llm.invoke(p.query)}

    return app
