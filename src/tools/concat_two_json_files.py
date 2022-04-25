import argparse
import json


def concat_two_json_files(file_1, file_2, output):
    """
    Load two json files and concatenate them.
    """

    with open(file_1, 'r') as f:
        data_1 = json.load(f)

    print(f"Loaded file 1 with {len(data_1)} samples")

    with open(file_2, 'r') as f:
        data_2 = json.load(f)

    print(f"Loaded file 2 with {len(data_2)} samples")

    data_1.extend(data_2)

    with open(output, 'w') as f:
        json.dump(data_1, f, indent=2)

    print(f"Saved concatenated file with {len(data_1)} samples to {output}")


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--file_1', type=str, default='data/spider/train.json')
    arg_parser.add_argument('--file_2', type=str, default='data/cordis/train_reduced.json')
    arg_parser.add_argument('--output', type=str, default='data/spider/dev_spider_cordis_with_synthetic_but_without_handmade.json')

    args = arg_parser.parse_args()

    concat_two_json_files(args.file_1, args.file_2, args.output)