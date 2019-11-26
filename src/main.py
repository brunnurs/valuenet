import datetime
import os

from src.config import read_arguments_train, write_config_to_file


def create_experiment_folder(model_output_dir, data_dir):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    experiment_name = "{}__{}".format(data_dir.upper(), timestamp)

    output_path = os.path.join(model_output_dir, experiment_name)
    os.makedirs(output_path, exist_ok=True)

    return experiment_name


if __name__ == '__main__':
    args = read_arguments_train()

    exp_name = create_experiment_folder(args.model_output_dir, args.data_dir)

    write_config_to_file(args, args.model_output_dir, exp_name)

