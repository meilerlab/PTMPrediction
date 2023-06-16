conda create -n predict_PTMs python=3.7.12
conda activate predict_PTMs
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
pip install --user tensorflow==2.8.1
conda install scikit-learn -c conda-forge
conda install -c conda-forge imbalanced-learn
conda install -c bioconda logomaker
conda install seaborn -c conda-forge
