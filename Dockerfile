# The vLLM Dockerfile is used to construct vLLM image that can be directly used
# to run the OpenAI compatible server.

# Example to build on 4xGH200 node.
# podman build --build-arg max_jobs=64 --build-arg nvcc_threads=8 --build-arg buildkite_commit=$(git rev-parse --short HEAD) --progress plain --target vllm-base --tag vllm:v0.6.0-$(git rev-parse --short HEAD)-arm64-cuda-gh200 .


# Please update any changes made here to
# docs/source/dev/dockerfile/dockerfile.rst and
# docs/source/assets/dev/dockerfile-stages-dependency.png

ARG CUDA_VERSION=12.5.0
#################### BASE BUILD IMAGE ####################
# prepare basic build environment
FROM nvcr.io/nvidia/pytorch:24.07-py3 AS base
#FROM nvcr.io/nvidia/pytorch:24.07-py3 AS base-reqs
ARG CUDA_VERSION=12.5.0
ARG PYTHON_VERSION=3.10
# cuda arch list used by torch
# can be useful for both `dev` and `test`
# explicitly set the list to avoid issues with torch 2.2
# see https://github.com/pytorch/pytorch/pull/123243
ARG torch_cuda_arch_list='9.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y \
    && apt-get install -y ccache software-properties-common git curl sudo vim libibverbs-dev

# Workaround for https://github.com/openai/triton/issues/2507 and
# https://github.com/pytorch/pytorch/issues/107960 -- hopefully
# this won't be needed for future versions of this docker image
# or future versions of triton.
#RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/

WORKDIR /workspace
# max jobs used by Ninja to build extensions
ARG max_jobs
ENV MAX_JOBS=${max_jobs}

# Install build and runtime dependencies from unlocked requirements
#COPY requirements-common.txt requirements-common.txt
#COPY requirements-cuda.txt requirements-cuda.txt
#RUN --mount=type=cache,target=/root/.cache/pip \
#    pip uninstall pynvml -y
#RUN --mount=type=cache,target=/root/.cache/pip \
#    pip install -r requirements-cuda.txt

# Install build and runtime dependencies from frozen requirements
COPY requirements-cuda-freeze.txt requirements-cuda-freeze.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip uninstall pynvml -y
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps -r requirements-cuda-freeze.txt

RUN mkdir vllm-aarch64-whl

RUN --mount=type=cache,target=/root/.cache/pip \
    pip --verbose wheel --use-pep517 --no-deps -w /workspace/vllm-aarch64-whl --no-build-isolation git+https://github.com/vllm-project/flash-attention.git@v2.6.2

RUN --mount=type=cache,target=/root/.cache/pip \
    pip --verbose wheel --use-pep517 --no-deps -w /workspace/vllm-aarch64-whl --no-build-isolation xformers==0.0.27

RUN --mount=type=cache,target=/root/.cache/pip \
    git clone https://github.com/openai/triton ; git checkout release/3.0.x ; cd triton/python ; git submodule update --init --recursive ;  pip --verbose wheel -w /workspace/vllm-aarch64-whl .

# Needed to ensure causal-conv1d builds from scratch
ENV CAUSAL_CONV1D_FORCE_BUILD=TRUE
ENV CAUSAL_CONV1D_SKIP_CUDA_BUILD=FALSE

RUN --mount=type=cache,target=/root/.cache/pip \
    pip --verbose wheel --use-pep517 --no-deps -w /workspace/vllm-aarch64-whl --no-build-isolation --no-cache-dir git+https://github.com/Dao-AILab/causal-conv1d.git@v1.4.0

ENV MAMBA_FORCE_BUILD=TRUE
RUN --mount=type=cache,target=/root/.cache/pip \
    pip --verbose wheel --use-pep517 --no-deps -w /workspace/vllm-aarch64-whl --no-build-isolation --no-cache-dir git+https://github.com/state-spaces/mamba.git@62db608da60f6fc790b8ed9f4b3225e95ca15fde

RUN --mount=type=cache,target=/root/.cache/pip \
	git clone https://github.com/flashinfer-ai/flashinfer.git ; git checkout v0.1.6 ; cd flashinfer/python ; pip --verbose wheel --use-pep517 --no-deps -w /workspace/vllm-aarch64-whl --no-build-isolation --no-cache-dir .

# This is a test.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install /workspace/vllm-aarch64-whl/*.whl --no-cache-dir --no-deps

#################### BASE BUILD IMAGE ####################

#################### WHEEL BUILD IMAGE ####################
FROM base AS build

# files and directories related to build wheels
COPY csrc csrc
COPY setup.py setup.py
COPY cmake cmake
COPY CMakeLists.txt CMakeLists.txt
COPY requirements-common.txt requirements-common.txt
COPY requirements-cuda.txt requirements-cuda.txt
COPY pyproject.toml pyproject.toml
COPY vllm vllm

# max jobs used by Ninja to build extensions
ARG max_jobs
ENV MAX_JOBS=${max_jobs}
# number of threads used by nvcc
ARG nvcc_threads
ENV NVCC_THREADS=$nvcc_threads

ARG buildkite_commit
ENV BUILDKITE_COMMIT=$buildkite_commit

ENV CCACHE_DIR=/root/.cache/ccache
RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/pip \
    python setup.py bdist_wheel --dist-dir=dist

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
#################### EXTENSION Build IMAGE ####################


#################### vLLM installation IMAGE ####################
# image with vLLM installed
FROM nvcr.io/nvidia/pytorch:24.07-py3 AS vllm-base
ARG CUDA_VERSION=12.5.0
ARG PYTHON_VERSION=3.10
WORKDIR /vllm-workspace
# max jobs used by Ninja to build extensions
ARG max_jobs
ENV MAX_JOBS=${max_jobs}

# cuda arch list used by torch
# can be useful for both `dev` and `test`
# explicitly set the list to avoid issues with torch 2.2
# see https://github.com/pytorch/pytorch/pull/123243
ARG torch_cuda_arch_list='9.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y \
    && apt-get install -y ccache software-properties-common git curl sudo vim libibverbs-dev

# Workaround for https://github.com/openai/triton/issues/2507 and
# https://github.com/pytorch/pytorch/issues/107960 -- hopefully
# this won't be needed for future versions of this docker image
# or future versions of triton.
#RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/

# Install build and runtime dependencies from unlocked requirements
#COPY requirements-common.txt requirements-common.txt
#COPY requirements-cuda.txt requirements-cuda.txt
#RUN --mount=type=cache,target=/root/.cache/pip \
#    pip uninstall pynvml -y
#RUN --mount=type=cache,target=/root/.cache/pip \
#    pip install -r requirements-cuda.txt

# Freeze the requirements, use this to update the requirements-cuda-freeze.txt to reproduce the same environment
RUN pip list --format freeze > /opt/requirements-cuda-freeze.txt

# Install build and runtime dependencies from frozen requirements
COPY requirements-cuda-freeze.txt requirements-cuda-freeze.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip uninstall pynvml -y
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps -r requirements-cuda-freeze.txt

RUN --mount=type=bind,from=base,src=/workspace/vllm-aarch64-whl,target=/workspace/vllm-aarch64-whl \
    --mount=type=cache,target=/root/.cache/pip \
    pip install /workspace/vllm-aarch64-whl/*.whl --no-deps --no-cache-dir

RUN --mount=type=bind,from=build,src=/workspace/dist,target=/vllm-workspace/dist \
    --mount=type=cache,target=/root/.cache/pip \
    pip install dist/*.whl --verbose

#################### vLLM installation IMAGE ####################

#################### OPENAI API SERVER ####################
# openai api server alternative
FROM vllm-base AS vllm-openai

# install additional dependencies for openai api server
# TODO Fix those versions.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install accelerate hf_transfer 'modelscope!=1.15.0'

ENV VLLM_USAGE_SOURCE production-docker-image

ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
#################### OPENAI API SERVER ####################
