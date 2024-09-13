# The vLLM Dockerfile is used to construct vLLM image that can be directly used
# to run the OpenAI compatible server.

# Example to build on 4xGH200 node.
# podman build --build-arg max_jobs=64 --build-arg nvcc_threads=8 --build-arg buildkite_commit=$(git rev-parse --short HEAD) --target vllm-base --tag vllm:v0.6.0-$(git rev-parse --short HEAD)-arm64-cuda-gh200 .
# Example to build on 24 core node.
# docker build --build-arg max_jobs=12 --build-arg nvcc_threads=4 --build-arg buildkite_commit=$(git rev-parse --short HEAD) --target vllm-base --tag vllm:v0.6.0-$(git rev-parse --short HEAD)-amd64-cuda-a100-h100 .


#################### BASE BUILD IMAGE ####################
# prepare basic build environment
FROM nvcr.io/nvidia/pytorch:24.07-py3 AS base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y \
    && apt-get install -y ccache software-properties-common git curl sudo vim libibverbs-dev

WORKDIR /workspace

ARG torch_cuda_arch_list='8.0 9.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
# max jobs used by Ninja to build extensions
ARG max_jobs
ENV MAX_JOBS=${max_jobs}
# number of threads used by nvcc
ARG nvcc_threads
ENV NVCC_THREADS=$nvcc_threads

COPY requirements-common.txt requirements-common.txt
COPY requirements-cuda.txt requirements-cuda.txt
RUN pip uninstall pynvml -y

# Install build and runtime dependencies from unlocked requirements
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-cuda.txt

RUN pip list --format freeze > /opt/requirements-cuda-freeze.txt

# Install build and runtime dependencies from frozen requirements
#COPY requirements-cuda-freeze.txt requirements-cuda-freeze.txt
#RUN --mount=type=cache,target=/root/.cache/pip \
#    pip install --no-deps -r requirements-cuda-freeze.txt

#################### BASE BUILD IMAGE ####################

#################### Build IMAGE ####################
FROM base AS build

# build vLLM extensions

RUN mkdir wheels

# vLLM flash-attention. flash-attn=2.4.2 is also included in the NGC PyTorch image.
# https://github.com/vllm-project/flash-attention/blob/344c988d1dcea878e65eb8cfb637eae621fc65b7/.github/workflows/scripts/build.sh#L16
RUN git clone https://github.com/vllm-project/flash-attention.git ; cd flash-attention ; git checkout v2.6.2 ; git submodule update --init --recursive ; python setup.py bdist_wheel --dist-dir=/workspace/wheels

# xFormers also install its flash-attention.
# https://github.com/facebookresearch/xformers/blob/d3948b5cb9a3711032a0ef0e036e809c7b08c1e0/.github/workflows/wheels_build.yml#L120
RUN git clone https://github.com/facebookresearch/xformers.git ; cd xformers ; git checkout v0.0.27 ; git submodule update --init --recursive ; python setup.py bdist_wheel --dist-dir=/workspace/wheels

# Flashinfer.
# https://github.com/flashinfer-ai/flashinfer/blob/8f186cf0ea07717727079d0c92bbe9be3814a9cb/scripts/run-ci-build-wheel.sh#L47C1-L47C39
RUN git clone https://github.com/flashinfer-ai/flashinfer.git ; cd flashinfer ; git checkout v0.1.6 ; git submodule update --init --recursive ; cd python ; python setup.py bdist_wheel --dist-dir=/workspace/wheels

# Install them.
RUN pip install --no-deps /workspace/wheels/*.whl

WORKDIR /vllm-workspace

# files and directories related to build wheels
COPY csrc csrc
COPY setup.py setup.py
COPY cmake cmake
COPY CMakeLists.txt CMakeLists.txt
COPY pyproject.toml pyproject.toml
COPY vllm vllm
COPY requirements-common.txt requirements-common.txt
COPY requirements-cuda.txt requirements-cuda.txt

ARG buildkite_commit
ENV BUILDKITE_COMMIT=$buildkite_commit

ENV CCACHE_DIR=/root/.cache/ccache
RUN --mount=type=cache,target=/root/.cache/ccache \
    python setup.py bdist_wheel --dist-dir=/workspace/wheels

# Check the size of the wheel if RUN_WHEEL_CHECK is true
COPY .buildkite/check-wheel-size.py check-wheel-size.py
# Default max size of the wheel is 250MB
ARG VLLM_MAX_SIZE_MB=250
ENV VLLM_MAX_SIZE_MB=$VLLM_MAX_SIZE_MB
ARG RUN_WHEEL_CHECK=true
RUN if [ "$RUN_WHEEL_CHECK" = "true" ]; then \
        python check-wheel-size.py dist; \
    else \
        echo "Skipping wheel size check."; \
    fi
####################  Build IMAGE ####################


#################### vLLM installation IMAGE ####################
# image with vLLM installed
FROM base AS vllm-base

RUN --mount=type=bind,from=build,src=/workspace/wheels,target=/workspace/wheels \
    pip install --no-deps /workspace/wheels/*.whl

#################### vLLM installation IMAGE ####################

#################### OPENAI API SERVER ####################
# openai api server alternative
FROM vllm-base AS vllm-openai

# install additional dependencies for openai api server
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install accelerate hf_transfer 'modelscope!=1.15.0'

# Freeze the requirements, use this to update the requirements-openai-freeze.txt to reproduce the same environment
RUN pip list --format freeze > /opt/requirements-openai-freeze.txt

# Install from freeze
#COPY requirements-openai-freeze.txt requirements-openai-freeze.txt
#RUN --mount=type=cache,target=/root/.cache/pip \
#    pip install --no-deps -r requirements-openai-freeze.txt

ENV VLLM_USAGE_SOURCE production-docker-image

ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
#################### OPENAI API SERVER ####################
