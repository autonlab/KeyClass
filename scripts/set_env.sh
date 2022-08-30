# Creates an environment to run experiments
ENV_NAME="keyclass" # Write the name of your environment 

echo "Creating environment ${ENV_NAME}..."
conda create -n $ENV_NAME python=3.7
conda activate $ENV_NAME
conda install -c conda-forge snorkel=0.9.8 sentence-transformers=2.0.0
conda install -c huggingface transformers=4.6.1 tokenizers=0.10.1
conda install -c anaconda jupyter
