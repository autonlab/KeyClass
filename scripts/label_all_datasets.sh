# Label all the datasets

DATA_PATH="/zfsauton/project/public/chufang/classes/"
# DATA_PATH = "../datasets/" # Uncomment

# DATASETS=('imdb' 'agnews' 'amazon' 'dbpedia')
DATASETS=('imdb')

MODEL_NAME="all-mpnet-base-v2"

for DATASET in "${DATASETS[@]}"
do
	echo "======================="
	echo "Labeling the training set for the $DATASET dataset..."
	python label_data.py --data_path=$DATA_PATH --dataset=$DATASET --model_name=$MODEL_NAME
done