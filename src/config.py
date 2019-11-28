import argparse
import json
import os


class Config:
    DATA_PREFIX = "data"
    EXPERIMENT_PREFIX = "experiments"

def write_config_to_file(args, model_output_dir: str, experiment_name: str):
    config_path = os.path.join(model_output_dir, experiment_name, "args.json")

    with open(config_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def read_arguments_train():
    parser = argparse.ArgumentParser(description="Run training with following arguments")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--toy', default=False, action='store_true')
    parser.add_argument('--data_set', default='spider', type=str)
    parser.add_argument('--batch_size_encoder', default=4, type=int)
    parser.add_argument('--encoder_pretrained_model', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_length', default=222, type=int)

    parser.add_argument('--num_epochs', default=5.0, type=float)

    parser.add_argument('--learning_rate', default=2e-5, type=float)
    parser.add_argument('--adam_eps', default=1e-8, type=float)
    parser.add_argument('--warmup_steps', default=0, type=int)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)

    args = parser.parse_args()

    args.data_dir = os.path.join(Config.DATA_PREFIX, args.data_set)
    args.model_output_dir = Config.EXPERIMENT_PREFIX

    print("*** parsed configuration from command line and combine with constants ***")

    for argument in vars(args):
        print("argument: {}={}".format(argument, getattr(args, argument)))

    return args
