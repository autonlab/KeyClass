# Label all the datasets

CONFIG="../config_coronary_angio_cabg.yml"
 
export CUDA_VISIBLE_DEVICES=3

echo "======================="
echo "Encoding the dataset..."
python encode_datasets.py --config=$CONFIG

echo "======================="
echo "Labeling the data..."
python label_data.py --config=$CONFIG

echo "======================="
echo "Training the downstream classifier..."
python train_downstream_model.py --config=$CONFIG