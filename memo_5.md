python3 -m pip list


環境構築
module purge 
module load python/3.10
module load cuda/12.6
module load cudnn/9.5.1.17
module load singularity
module list
export CC=gcc
export CXX=g++

export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="9.0"

cd rllm
singularity build --fakeroot pytorch-24.12.sif pytorch-24.12.def

singularity run --nv pytorch-24.12.sif

cd verl
MAX_JOBS=4 python -m pip install -e . --no-build-isolation -v
cd rllm
python -m pip install -e . -v


chmod +777 ./scripts/deepscaler/train/deepscaler_1.5b_test.sh
export MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
./scripts/deepscaler/train/deepscaler_1.5b_test.sh --model $MODEL_PATH



export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="9.0"


uv venv rllm_env_5 --python 3.12
source rllm_env_5/bin/activate
uv pip install --upgrade pip
uv pip install setuptools
uv pip install ninja

uv pip install torch==2.6.0 torchaudio torchvision --index-url https://download.pytorch.org/whl/cu126

mkdir -p ./packages
curl -o ./packages/flash_attn-2.7.2.post1-cp312-cp312-linux_aarch64.whl https://static.abacus.ai/pypi/abacusai/gh200-llm/pytorch-2412-cuda126/flash_attn-2.7.2.post1-cp312-cp312-linux_aarch64.whl
curl -o ./packages/xformers-0.0.30+46a02df6.d20250103-cp312-cp312-linux_aarch64.whl https://static.abacus.ai/pypi/abacusai/gh200-llm/pytorch-2412-cuda126/xformers-0.0.30%2B46a02df6.d20250103-cp312-cp312-linux_aarch64.whl
curl -o ./packages/megablocks-0.7.0-cp312-cp312-linux_aarch64.whl https://static.abacus.ai/pypi/abacusai/gh200-llm/pytorch-2412-cuda126/megablocks-0.7.0-cp312-cp312-linux_aarch64.whl
curl -o ./packages/bitsandbytes-0.45.1.dev0-cp312-cp312-linux_aarch64.whl https://static.abacus.ai/pypi/abacusai/gh200-llm/pytorch-2412-cuda126/bitsandbytes-0.45.1.dev0-cp312-cp312-linux_aarch64.whl
curl -o ./packages/vllm-0.7.3+cu126-cp312-cp312-linux_aarch64.whl https://static.abacus.ai/pypi/abacusai/gh200-llm/pytorch-2412-cuda126/vllm-0.7.3%2Bcu126-cp312-cp312-linux_aarch64.whl
curl -o ./packages/decord-0.6.0-cp312-cp312-linux_aarch64.whl https://static.abacus.ai/pypi/abacusai/gh200-llm/pytorch-2412-cuda126/decord-0.6.0-patched-cp312-cp312-linux_aarch64.whl
curl -o ./packages/sglang-0.4.3.post2-py3-none-any.whl https://static.abacus.ai/pypi/abacusai/gh200-llm/pytorch-2412-cuda126/sglang-0.4.3.post2-py3-none-any.whl
curl -o ./packages/sglang_router-0.1.4-cp312-cp312-linux_aarch64.whl https://static.abacus.ai/pypi/abacusai/gh200-llm/pytorch-2412-cuda126/sglang_router-0.1.4-cp312-cp312-linux_aarch64.whl
curl -o ./packages/sgl_kernel-0.0.3.post6-cp39-abi3-manylinux2014_aarch64.whl https://static.abacus.ai/pypi/abacusai/gh200-llm/pytorch-2412-cuda126/sgl_kernel-0.0.3.post6-cp39-abi3-manylinux2014_aarch64.whl
curl -o ./packages/flashinfer_python-0.2.1.post2-cp38-abi3-linux_aarch64.whl https://static.abacus.ai/pypi/abacusai/gh200-llm/pytorch-2412-cuda126/flashinfer_python-0.2.1.post2-cp38-abi3-linux_aarch64.whl

uv pip install --no-deps --no-index ./packages/*.whl


git clone https://github.com/puwaer/rllm.git
cd verl
MAX_JOBS=4 uv pip install -e . --no-build-isolation -v