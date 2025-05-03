deactivate
rm -rf rllm_env
rm -rf /work/gj26/j26001/.cache/uv
rm -rf ~/.cache/uv

python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
python -c "import flash_attn; print(getattr(flash_attn, '__version__', 'No version info'))"
python -c "import vllm; print('vLLM version:', getattr(vllm, '__version__', 'Unknown')); from vllm import LLM; print('LLM class loaded successfully')"

find ~/.cache/pip -name "vllm*.whl"
find ~/.cache/uv -name "vllm*.whl"
find ~/.cache/uv -name "triton*.whl"
find ~/.cache/uv -name "bitsandbytes*.whl"
find ~/.cache/uv -name "flash_attn*.whl"
find ~/.cache/uv -name "verl*.whl"


uv pip install /work/gj26/j26001/wheels/vllm-0.7.4.dev0+ged6e9075d.d20250502.cu124-0.editable-cp310-cp310-linux_aarch64.whl
uv pip install /work/gj26/j26001/wheels/vllm-0.6.4.dev0+gfd47e57f4.d20250503.cu124-0.editable-cp310-cp310-linux_aarch64.whl
uv pip install /work/gj26/j26001/wheels/triton-3.1.0-0.editable-cp310-cp310-linux_aarch64.whl
uv pip install /work/gj26/j26001/wheels/bitsandbytes-0.45.5-0.editable-cp310-cp310-linux_aarch64.whl
uv pip install /work/gj26/j26001/wheels/flash_attn-2.7.4.post1-cp310-cp310-linux_aarch64.whl


qsub -I -l select=1 -W group_list=gj26 -q interact-g -l walltime=02:00:00

環境構築
module purge 
module load python/3.10
module load cuda/12.4
module load cudnn/9.5.1.17
module list
export CC=gcc
export CXX=g++

export CUDA_VISIBLE_DEVICES=0


uv venv rllm_env --python 3.10
source rllm_env/bin/activate
uv pip install --upgrade pip
uv pip install setuptools


uv pip install torch==2.4.0 torchaudio torchvision --index-url https://download.pytorch.org/whl/cu124


git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.6.3
python use_existing_torch.py
uv pip install -r requirements-build.txt
MAX_JOBS=4 uv pip install -e . --no-build-isolation -v


安定版 0.3.1, 0.4.2, 0.5.4, 0.6.3


git clone https://github.com/triton-lang/triton.git
cd triton
git checkout remotes/origin/release/3.1.x
uv pip install psutil
uv pip install -e python -v
uv pip install /work/gj26/j26001/wheels/triton-3.1.0-0.editable-cp310-cp310-linux_aarch64.whl


git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git
cd bitsandbytes/
git checkout 0.45.5
cmake -DCOMPUTE_BACKEND=cuda -S
make
uv pip install -e . -v
uv pip install /work/gj26/j26001/wheels/bitsandbytes-0.45.5-0.editable-cp310-cp310-linux_aarch64.whl



export TORCH_CUDA_ARCH_LIST="9.0"
MAX_JOBS=4 uv pip install flash-attn==flash-attn==2.7.4.post1 --no-build-isolation -v


git clone https://github.com/volcengine/verl.git
cd verl
uv pip install -e . -v



git clone https://github.com/agentica-project/rllm.git
cd rllm
pip install -e ./verl
pip install -e . -v

cd rllm/verl
MAX_JOBS=4 uv pip install -e . --no-build-isolation -v
cd rllm
pip install -e . -v


python scripts/data/download_datasets.py
python scripts/data/deepcoder_dataset.py

huggingface-cli login
wandb login

chmod +777 ./scripts/deepscaler/train/deepscaler_1.5b_8k.sh
export MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
./scripts/deepscaler/train/deepscaler_1.5b_8k.sh --model $MODEL_PATH
