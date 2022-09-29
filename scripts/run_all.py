import argparse
import label_data, encode_datasets, train_downstream_model

if __name__ == "__main__":
    parser_cmd = argparse.ArgumentParser()
    parser_cmd.add_argument('--config',
                            default='../config_files/config_dbpedia.yml',
                            help='Configuration file')
    parser_cmd.add_argument('--random_seed',
                            default=0,
                            type=int,
                            help="Random Seed")
    args_cmd = parser_cmd.parse_args()

    print("Encoding Dataset")
    encode_datasets.run(args_cmd)

    print("Labeling Data")
    label_data.run(args_cmd)

    print("Training Model")
    results = train_downstream_model.train(args_cmd)
    print("Model Results:")
    print(results)
