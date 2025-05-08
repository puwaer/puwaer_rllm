conda環境構築
conda create -n tiny_zero python=3.9
conda activate tiny_zero

pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install vllm==0.6.3 
pip3 install ray

cd Document/puwaer_code_trans/TinyZero
pip install -e .

pip3 install flash-attn --no-build-isolation
pip install wandb IPython matplotlib


実行コマンド
conda activate tiny_zero


python ./examples/data_preprocess/countdown.py --local_dir ./dataset


学習コマンド
conda activate tiny_zero

cd Document/puwaer_code_trans/TinyZero
cd TinyZero

export N_GPUS=1
export BASE_MODEL=base_model/Qwen2.5-1.5B-Instruct
export DATA_DIR=dataset
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=countdown-qwen2.5-1.5b_test_2
export VLLM_ATTENTION_BACKEND=XFORMERS

chmod +777 ./scripts/train_tiny_zero_qwen_1.5b.sh
bash ./scripts/train_tiny_zero_qwen_1.5b.sh



miyabi環境構築
deactivate

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


uv venv rllm_env_5 --python 3.10
source rllm_env_5/bin/activate
uv pip install --upgrade pip
uv pip install setuptools
uv pip install ninja

uv pip install torch==2.6.0 torchaudio torchvision --index-url https://download.pytorch.org/whl/cu126


uv pip install /work/gj26/j26001/wheels/vllm-0.6.3+cu126-0.editable-cp310-cp310-linux_aarch64.whl
uv pip install /work/gj26/j26001/wheels/triton-3.1.0-0.editable-cp310-cp310-linux_aarch64.whl
uv pip install /work/gj26/j26001/wheels/bitsandbytes-0.45.5-0.editable-cp310-cp310-linux_aarch64.whl
uv pip install /work/gj26/j26001/wheels/xformers-0.0.28+c909f0d6.d20250506-0.editable-cp310-cp310-linux_aarch64.whl
uv pip install /work/gj26/j26001/wheels/flash_attn-2.7.4.post1-cp310-cp310-linux_aarch64.whl


cd puwaer_code_trans/TinyZero
pip install -e .

uv pip install wandb IPython matplotlib


miyabi実行コマンド
python ./examples/data_preprocess/countdown.py --local_dir ./dataset/countdown

export N_GPUS=1
export BASE_MODEL=base_model/Qwen2.5-1.5B-Instruct
export DATA_DIR=dataset/countdown
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=countdown-qwen2.5-1.5b_test_miyabi_1
export VLLM_ATTENTION_BACKEND=XFORMERS

chmod +777 ./scripts/train_tiny_zero_qwen_1.5b.sh
bash ./scripts/train_tiny_zero_qwen_1.5b.sh