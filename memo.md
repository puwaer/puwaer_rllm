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

conda create -n test_rllm python=3.11
conda activate test_rllm

pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

pip install packaging
pip install -U pip setuptools packaging wheel ninja pybind11

cd /Document/puwaer_rllm/flash-attention/hopper
MAX_JOBS=5 pip install .
MAX_JOBS=5 python setup.py install

これを使用した
MAX_JOBS=4 pip install flash-attn --no-build-isolation



git clone https://github.com/vllm-project/vllm.git
cd vllm
python use_existing_torch.py
pip install -r requirements/build.txt
MAX_JOBS=4 pip install --no-build-isolation -e .

pip install .
