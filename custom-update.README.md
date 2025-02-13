```bash
git remote add vllm-remote git@github.com:vllm-project/vllm.git
git fetch --tags vllm-remote

git checkout <old-version>
git checkout -b freeze-old-version

git checkout <new-version>
git checkout -b freeze-new-version

# Diff with PyCharm
# left click -> git -> compare with branch -> freeze-old-version
# on these files

# requirements-build.txt
# requirements-common.txt
# requirements-cuda.txt
# dockerfile

# Update the files accordingly in main of vllm-build

git checkout -b custom-build-<new-version>
# Copy the files mentioned aboveq

# on the CSCS cluster with a 4xGH200 node.
git tag <new-version>
podman build --build-arg max_jobs=64 --build-arg nvcc_threads=8  --target vllm-base --tag vllm:<new-version>-$(git rev-parse --short HEAD)-arm64-cuda-gh200 .

```
