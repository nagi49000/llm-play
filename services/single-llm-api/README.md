# Service single_llm_api

Build a containerized FastAPI app wrapping a LLM.

For CPU only support, build and run with
```bash
docker build --build-arg pip_cuda_index_url=https://download.pytorch.org/whl/cpu --build-arg githash=$(git rev-parse HEAD) --build-arg build_time=$(date --utc --iso-8601='seconds') . -t model-cpu
docker run --rm -p 26780:6780 model-cpu
```

For GPU support, build and run with
```bash
docker build --build-arg pip_cuda_index_url=https://download.pytorch.org/whl/cu118 --build-arg githash=$(git rev-parse HEAD) --build-arg build_time=$(date --utc --iso-8601='seconds') . -t model-gpu
docker run --gpus all --rm -p 26780:6780 model-gpu
```
Note that the build arg pip_cuda_index_url should point to a repo corresponding to the major CUDA version for the prod deployment. Based onthe CUDA enviroment, one can work out the index-url from [here](https://pytorch.org/get-started/locally/ ). Using GPUs with docker (build and run) will require the [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

The LLM is baked into the docker image. The params for the LLM that is downloaded and baked in are specified in [this yaml file](src/simple_llm_api/llm-pipeline.yaml).

After build, the image should be self-contained (i.e. will not _need_, although may try, to download further LLM files or packages) and only need CPU for inference. When starting the image in a container without the internet, the app may take a couple of minutes to start up whilst the calls to https://huggingface.co for more up to date json configs on models, tokenizers etc times out.

Once the app is up, swagger docs should be available at `http://localhost:26780/docs`.
