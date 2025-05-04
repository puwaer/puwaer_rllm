deactivate
rm -rf rllm_env_4
rm -rf /work/gj26/j26001/.cache/uv
rm -rf ~/.cache/uv

python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
python -c "import flash_attn; print(getattr(flash_attn, '__version__', 'No version info'))"
python -c "import vllm; print('vLLM version:', getattr(vllm, '__version__', 'Unknown')); from vllm import LLM; print('LLM class loaded successfully')"
python -c "import vllm; print(vllm.__version__)"
python -c "import torch; print(torch.__version__); print(hasattr(torch.backends.cuda, 'is_flash_attention_available'))"



find ~/.cache/pip -name "vllm*.whl"
find ~/.cache/uv -name "vllm*.whl"
find ~/.cache/uv -name "triton*.whl"
find ~/.cache/uv -name "bitsandbytes*.whl"
find ~/.cache/uv -name "flash_attn*.whl"
find ~/.cache/uv -name "verl*.whl"
find ~/.cache/uv -name "xformers*.whl"


git log -1



uv pip install /work/gj26/j26001/wheels/xformers-0.0.31+8fc8ec5a.d20250504-0.editable-cp310-cp310-linux_aarch64.whl
uv pip install /work/gj26/j26001/wheels/
uv pip install /work/gj26/j26001/wheels/
uv pip install /work/gj26/j26001/wheels/



qsub -I -l select=1 -W group_list=gj26 -q interact-g -l walltime=02:00:00

環境構築
module purge 
module load python/3.10
module load cuda/12.6
module load cudnn/9.5.1.17
module list
export CC=gcc
export CXX=g++

export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="9.0"


uv venv rllm_env_4 --python 3.10
source rllm_env_4/bin/activate
uv pip install --upgrade pip
uv pip install setuptools


pip install ninja

uv pip install torch==2.6.0 torchaudio torchvision --index-url https://download.pytorch.org/whl/cu126


git clone https://github.com/facebookresearch/xformers.git
cd xformers
git submodule update --init --recursive
MAX_JOBS=4 uv pip install -e . --no-build-isolation -v


git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.6.3
python use_existing_torch.py
uv pip install -r requirements-build.txt
export SETUPTOOLS_SCM_PRETEND_VERSION=0.6.3
MAX_JOBS=4 uv pip install -e . --no-build-isolation -v

git clone https://github.com/triton-lang/triton.git
cd triton
git checkout remotes/origin/release/3.1.x
uv pip install psutil
uv pip install -e python -v


git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git
cd bitsandbytes/
git checkout 0.45.5
cmake -DCOMPUTE_BACKEND=cuda -S
make
uv pip install -e . -v
