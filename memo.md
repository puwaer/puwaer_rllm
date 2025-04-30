conda clean --all
conda create -n test_rllm python=3.11
conda activate test_rllm

conda deactivate
conda env remove --name test_rllm


qsub -I -l select=1 -W group_list=gj26 -q interact-g -l walltime=01:00:00

module purge 
module load python/3.11
module load singularity/4.2.1
module load cuda/12.4
module list


環境構築コマンド
module purge 
module load python/3.11
module load cuda/12.4

conda create -n test_rllm python=3.11
conda activate test_rllm



python==3.11
torch==2.5.1+cu124