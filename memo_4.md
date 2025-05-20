deactivate
rm -rf rllm_env_4
rm -rf /work/gj26/j26001/.cache/uv
rm -rf ~/.cache/uv

python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
python -c "import torch; print(torch.__version__); print(hasattr(torch.backends.cuda, 'is_flash_attention_available'))"
python -c "import flash_attn; print(getattr(flash_attn, '__version__', 'No version info'))"
python -c "import vllm; print('vLLM version:', getattr(vllm, '__version__', 'Unknown')); from vllm import LLM; print('LLM class loaded successfully')"
python -c "import vllm; print(vllm.__version__)"
python -c "import apex; print('Apex installed successfully')"
python -c "import torch; from apex import amp; model = torch.nn.Linear(10, 10).cuda(); optimizer = torch.optim.SGD(model.parameters(), lr=0.01); model, optimizer = amp.initialize(model, optimizer, opt_level='O1'); print('Apex AMP initialized successfully')"
python -c "import apex; print('apex imported successfully')"
python -c "import triton; print(triton.__version__)"


find ~/.cache/uv -name "vllm*.whl"
find ~/.cache/uv -name "triton*.whl"
find ~/.cache/uv -name "bitsandbytes*.whl"
find ~/.cache/uv -name "flash_attn*.whl"
find ~/.cache/uv -name "verl*.whl"
find ~/.cache/uv -name "xformers*.whl"
find ~/.cache/uv -name "*.whl"
ls /work/gj26/j26001/wheels/

git log -1


uv pip install /work/gj26/j26001/wheels/vllm-0.6.3+cu126-0.editable-cp310-cp310-linux_aarch64.whl
uv pip install /work/gj26/j26001/wheels/triton-3.1.0-0.editable-cp310-cp310-linux_aarch64.whl
uv pip install /work/gj26/j26001/wheels/bitsandbytes-0.45.5-0.editable-cp310-cp310-linux_aarch64.whl

uv pip install /work/gj26/j26001/wheels/xformers-0.0.28+c909f0d6.d20250506-0.editable-cp310-cp310-linux_aarch64.whl
uv pip install /work/gj26/j26001/wheels/flash_attn-2.7.4.post1-cp310-cp310-linux_aarch64.whl
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
export CUDA_LAUNCH_BLOCKING=1

source rllm_env_4/bin/activate


uv venv rllm_env_4 --python 3.10
source rllm_env_4/bin/activate
uv pip install --upgrade pip
uv pip install setuptools
uv pip install ninja

uv pip install torch==2.6.0 torchaudio torchvision --index-url https://download.pytorch.org/whl/cu126


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
cd bitsandbytes
git checkout 0.45.5
cmake -DCOMPUTE_BACKEND=cuda -S
make
uv pip install -e . -v

git clone https://github.com/facebookresearch/xformers.git
cd xformers
git submodule update --init --recursive
MAX_JOBS=4 uv pip install -e . --no-build-isolation -v



git clone https://github.com/facebookresearch/xformers.git
cd xformers
git checkout v0.0.28
git submodule update --init --recursive
export CUTE_ARCH_MMA_SM90A_ENABLED=1
export CUDA_ARCHITECTURES=90 
export TORCH_CUDA_ARCH_LIST="9.0"
MAX_JOBS=4 uv pip install -e . --no-build-isolation -v



git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.7.4.post1
cd hopper
export CMAKE_ARGS="-DCMAKE_CUDA_FLAGS='-DCUTE_ARCH_MMA_SM90A_ENABLED'"
export TORCH_CUDA_ARCH_LIST="9.0"
MAX_JOBS=4 uv pip install -e . --no-build-isolation -v

MAX_JOBS=4 uv pip install flash-attn==2.7.4.post1 --no-build-isolation -v


git clone https://github.com/puwaer/rllm.git
cd rllm/verl
MAX_JOBS=4 uv pip install -e . --no-build-isolation -v
cd rllm
uv pip install -e . -v


python scripts/data/download_datasets.py
python scripts/data/deepscaler_dataset.py

huggingface-cli login
wandb login


chmod +777 ./scripts/deepscaler/train/deepscaler_1.5b_8k.sh
export MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
./scripts/deepscaler/train/deepscaler_1.5b_8k.sh --model $MODEL_PATH


cd rllm
chmod +777 ./scripts/deepscaler/train/deepscaler_1.5b_test.sh
export MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
./scripts/deepscaler/train/deepscaler_1.5b_test.sh --model $MODEL_PATH





実行コマンド
conda activate tiny_zero
python scripts/data/tiny_zero_countdown.py --local_dir ./dataset/countdown

学習コマンド
conda activate tiny_zero

cd Document/puwaer_code_trans/TinyZero
cd TinyZero

export N_GPUS=1
export BASE_MODEL=base_model/Qwen2.5-1.5B-Instruct
export DATA_DIR=data/countdown
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=countdown-qwen2.5-test3
export VLLM_ATTENTION_BACKEND=XFORMERS

chmod +777 ./scripts/tiny_zero/train_tiny_zero_grpo.sh
./scripts/tiny_zero/train_tiny_zero_grpo.sh