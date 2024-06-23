FROM registry.fedoraproject.org/f35/python3

COPY --chown=default:root requirements.txt requirements.txt

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY --chown=default:root src src
COPY --chown=default:root test_model_cpu test_model_cpu

RUN python -m pytest --cov=./src/ --cov-report term-missing

ARG githash=unset
ARG build_time=unset

ENV BUILD_TIME=${build_time}
ENV GITHASH=${githash}
ENV LOG_LEVEL=DEBUG

CMD gunicorn -k uvicorn.workers.UvicornWorker src.model_cpu.app:app --bind 0.0.0.0:6780 --log-level ${LOG_LEVEL}