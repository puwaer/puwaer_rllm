その他

conda deactivate 
conda env remove --name rllm -y

python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
python -c "import flash_attn; print(getattr(flash_attn, '__version__', 'No version info'))"


qsub -I -l select=1 -W group_list=gj26 -q interact-g -l walltime=02:00:00

環境構築
conda create -n rllm python=3.10 -y
conda activate rllm


module purge 
module load python/3.10
module load cuda/12.4
module load cudnn/9.5.1.17
module list
export CC=gcc
export CXX=g++

export CUDA_VISIBLE_DEVICES=0

pip install --upgrade pip
pip install setuptools

pip install torch==2.4.0 torchaudio torchvision --index-url https://download.pytorch.org/whl/cu124

pip install packaging>=20
pip install tomli
pip install ninja
pip install einops

export TORCH_CUDA_ARCH_LIST="9.0"
MAX_JOBS=4 pip install flash-attn==2.7.4.post1 --no-build-isolation -v


git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.7.3
python use_existing_torch.py 
pip install -r requirements-build.txt
MAX_JOBS=4 pip install -e . --no-build-isolation -v

git clone https://github.com/triton-lang/triton.git
cd triton
git checkout remotes/origin/release/3.1.x
pip install psutil
pip install -e python -v

git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git
cd bitsandbytes/
git checkout 0.45.5
cmake -DCOMPUTE_BACKEND=cuda -S . -B build
make
pip install -e . -v


git clone https://github.com/agentica-project/rllm.git
cd rllm
pip install -e ./verl
pip install -e .

python scripts/data/download_datasets.py
python scripts/data/deepcoder_dataset.py

chmod +777 ./scripts/deepscaler/train/deepscaler_1.5b_8k.sh
export MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
./scripts/deepscaler/train/deepscaler_1.5b_8k.sh --model $MODEL_PATH