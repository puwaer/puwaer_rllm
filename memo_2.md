その他

conda deactivate 
conda env remove --name rllm -y

python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"


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


MAX_JOBS=4 pip install flash-attn== --no-build-isolation -v


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

