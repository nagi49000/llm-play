FROM registry.fedoraproject.org/f35/python3

ARG githash=unset
ARG build_time=unset

LABEL githash=${githash}
LABEL build-time=${build_time}

COPY --chown=default:root requirements.txt requirements.txt
COPY --chown=default:root requirements_torch_only.txt requirements_torch_only.txt

RUN pip install --upgrade pip && \
    # seperate pip install repo for torch for CPU only support without
    # all the CUDA libs as described on https://pytorch.org/get-started/locally/
    pip install -r requirements_torch_only.txt --index-url https://download.pytorch.org/whl/cpu && \
    # rest of the requirements from normal pypi
    pip install -r requirements.txt

COPY --chown=default:root src src
COPY --chown=default:root test_model_cpu test_model_cpu

# Rather than run in a different build stage, run tests directly in this docker-build stage
# so that the LLM is downloaded and baked into the image
RUN python -m pytest -s -v --cov=./src/ --cov-report term-missing

ENV BUILD_TIME=${build_time}
ENV GITHASH=${githash}
ENV LOG_LEVEL=DEBUG

CMD gunicorn -k uvicorn.workers.UvicornWorker src.model_cpu.app:app --bind 0.0.0.0:6780 --log-level ${LOG_LEVEL}
