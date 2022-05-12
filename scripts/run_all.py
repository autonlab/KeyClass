import argparse
import label_data, encode_datasets, train_downstream_model
from mlflow import log_metric, log_param, log_artifacts
import mlflow.pytorch

def main(args):
    
    print("Labeling Data")
    label_data.run(args)
    print("Encoding Dataset")
    encode_datasets.run(args)
    print("Training Model")
    results = train_downstream_model.run(args)
    print("Model Results:")

    print(results)
    mlflow.log_metrics('results[0,0])

if __name__ == "__main__":
    parser_cmd = argparse.ArgumentParser()
    parser_cmd.add_argument('--config', default='../default_config.yml', help='Configuration file')
    parser_cmd.add_argument('--random_seed',default=0,help="Random Seed")
    args = parser_cmd.parse_args()

    main(args)
