# Service model-cpu

Build a containerized FastAPI app wrapping a LLM.

Build and run with
```
docker build --build-arg githash=$(git rev-parse HEAD) --build-arg build_time=$(date --utc --iso-8601='seconds') . -t model-cpu
docker run --rm -p 26780:6780 model-cpu
```
and swagger docs should be available at `http://localhost:26780/docs`

The LLM is baked into the docker image. The params for the LLM that is downloaded and baked in are specified in [this yaml file](src/model_cpu/llm-pipeline.yaml).

After build, the image should be self-contained (i.e. will not _need_, although may try, to download further LLM files or packages) and only need CPU for inference. When starting the image in a container without the internet, the app may take a couple of minutes to start up whilst the calls to https://huggingface.co for more up to data json configs on models, tokenizers etc times out.