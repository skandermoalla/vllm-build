```bash
git remote add vllm-remote git@github.com:vllm-project/vllm.git
git fetch --tags vllm-remote

git checkout old-version
git checkout -b freeze-old-version

git checkout new-version
git checkout -b freeze-new-version

# Diff with PyCharm
# left click -> git -> compare with branch -> freeze-old-version
# on these files

# requirements-dev.txt
# requirements-common.txt
# requirements-cuda.txt
# dockerfile
```
