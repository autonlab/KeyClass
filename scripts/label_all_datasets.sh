# Label all the datasets

# Encode all datasets

DATA_PATH="/zfsauton/project/public/chufang/classes/"
# DATA_PATH = "../datasets/" # Uncomment

# DATASETS=('imdb' 'agnews' 'amazon' 'dbpedia')
DATASETS=('imdb')

MODEL_NAME="all-mpnet-base-v2"

for DATASET in "${DATASETS[@]}"
do
	echo "======================="
	echo "Creating and saving embeddings for the $DATASET dataset"
	python encode_datasets.py --data_path=$DATA_PATH --dataset=$DATASET --model_name=$MODEL_NAME
done