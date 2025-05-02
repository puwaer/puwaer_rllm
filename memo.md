conda clean --all
conda create -n test_rllm python=3.11
conda activate test_rllm

conda deactivate
conda env remove --name test_rllm


qsub -I -l select=1 -W group_list=gj26 -q interact-g -l walltime=01:00:00
qsub -I -l select=1 -W group_list=gj26 -q interact-g -l walltime=02:00:00


module purge 
module load python/3.10
module load singularity/4.2.1
module load cuda/12.4
module list


環境構築コマンド
module purge 
module load cuda/12.4
module load cudnn/9.5.1.17
module list
export CC=gcc
export CXX=g++

export CUDA_VISIBLE_DEVICES=0

conda create -n test_rllm python=3.11
conda activate test_rllm

pip install --upgrade pip
pip install setuptools

pip install torch==2.5.1 torchaudio torchvision --index-url https://download.pytorch.org/whl/cu124

pip install packaging==25.0
pip install ninja==1.11.1.4
pip install einops==0.8.1

これを使用した
export TORCH_CUDA_ARCH_LIST="9.0"
FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
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



git clone https://github.com/volcengine/verl.git
cd verl
pip install -e . -v


git clone https://github.com/agentica-project/rllm.git
cd rllm
pip install -e . -v


実行コマンド
pwd /work/gj26/j26001/Document/puwaer_rllm/rllm
python scripts/data/download_datasets.py
python scripts/data/deepscaler_dataset.py

chmod +777 ./scripts/deepscaler/train/deepscaler_1.5b_8k.sh
export MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
./scripts/deepscaler/train/deepscaler_1.5b_8k.sh --model $MODEL_PATH


