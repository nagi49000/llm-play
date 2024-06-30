# use a Red Hat UBI image ready for kube deploy
FROM registry.fedoraproject.org/f35/python3

ARG githash=unset
ARG build_time=unset
# these URLs are used by fastapi to download rendering files for swagger and redoc docs
# they are working for fastapi==0.111.0, and may need to be updated for other versions of fastapi
ARG swagger_css_url=https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js
ARG swagger_favicon_url=https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui.css
ARG redoc_js_url=https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js
# default args for CPU only
ARG pip_cuda_index_url=https://download.pytorch.org/whl/cpu
ARG app_name="self contained CPU model"

LABEL githash=${githash}
LABEL build-time=${build_time}

COPY --chown=default:root requirements.txt requirements.txt
COPY --chown=default:root requirements_torch_only.txt requirements_torch_only.txt

RUN pip install --upgrade pip && \
    # seperate pip install repo for torch for CPU only support or the specific
    # GPU CUDA repo that marries up the with major CUDA version used on the deployment host
    # change index url to match CUDA version available on your host/deployment environment
    # all the CUDA libs as described on https://pytorch.org/get-started/locally/
    pip install -r requirements_torch_only.txt --index-url ${pip_cuda_index_url} && \
    # rest of the requirements from normal pypi
    pip install -r requirements.txt && \
    # download static assets for offline usage of fastapi docs UI
    mkdir static && \
    curl --output static/swagger-ui-bundle.css ${swagger_css_url} && \
    curl --output static/swagger-ui.css ${swagger_favicon_url} && \
    curl --output static/redoc.standalone.js ${redoc_js_url}

COPY --chown=default:root src src
COPY --chown=default:root test_single_llm_api test_single_llm_api

# Rather than run in a different build stage, run tests directly in this docker-build stage
# so that the LLM is downloaded and baked into the image
RUN python -m pytest -s -v --cov=./src/ --cov-report term-missing

ENV BUILD_TIME=${build_time}
ENV GITHASH=${githash}
ENV APP_NAME=${app_name}
ENV LOG_LEVEL=DEBUG

CMD gunicorn --worker-class uvicorn.workers.UvicornWorker --workers=1 --bind 0.0.0.0:6780 --log-level ${LOG_LEVEL} src.single_llm_api.app:app
