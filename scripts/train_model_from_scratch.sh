# Label all the datasets

# CONFIG="/zfsauton2/home/mgoswami/KeyClass/config_coronary_angio_cabg.yaml" # Add path to your config file
CONFIG="../config_coronary_angio_cabg.yaml" # Add path to your config file
 
export CUDA_VISIBLE_DEVICES=3 # Comment if you want to run on all the GPUs. 

echo "Encoding the dataset..."
python encode_datasets.py --config=$CONFIG
echo "======================="

echo "Labeling the data..."
python label_data.py --config=$CONFIG
echo "======================="

echo "Training the downstream classifier..."
python train_downstream_model.py --config=$CONFIG
echo "======================="