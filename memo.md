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
module list

conda create -n test_rllm python=3.11
conda activate test_rllm

pip install torch==2.5.1 torchaudio torchvision --index-url https://download.pytorch.org/whl/cu124

pip install packaging==25.0
pip install ninja==1.11.1.4
pip install einops==0.8.1

これを使用した
export TORCH_CUDA_ARCH_LIST="9.0"
FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
MAX_JOBS=4 pip install flash-attn==2.7.4.post1 --no-build-isolation -v

pip install vllm --no-build-isolation -v

