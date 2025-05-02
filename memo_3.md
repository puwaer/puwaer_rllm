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
git checkout v0.7.3
python use_existing_torch.py
uv pip install -r requirements-build.txt
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

export TORCH_CUDA_ARCH_LIST="9.0"
MAX_JOBS=4 pip install flash-attn --no-build-isolation -v
