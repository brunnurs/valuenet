import argparse
import json
import os


class Config:
    DATA_PREFIX = "data"
    EXPERIMENT_PREFIX = "experiments"

    # MODEL_CLASSES = {
    #     'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    #     'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    #     'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    #     'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    #     'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    # }


def write_config_to_file(args, model_output_dir: str, experiment_name: str):
    config_path = os.path.join(model_output_dir, experiment_name, "args.json")

    with open(config_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def read_arguments_train():
    parser = argparse.ArgumentParser(description="Run training with following arguments")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--data_dir', default='wikisql', type=str)

    args = parser.parse_args()

    args.data_path = os.path.join(Config.DATA_PREFIX, args.data_dir)
    args.model_output_dir = Config.EXPERIMENT_PREFIX

    print("*** parsed configuration from command line and combine with constants ***")

    for argument in vars(args):
        print("argument: {}={}".format(argument, getattr(args, argument)))

    return args
