import argparse
import json
import random


def pick_siblings(file_input, file_output_1, file_output_2, n_samples):

    with open(file_input, 'r') as f:
        input_data = json.load(f)

    ouput_data_handpicked = []

    initial_size = len(input_data)
    print(f"Loaded input file with {len(input_data)} samples")

    for idx in range(n_samples):
        element_to_pick = random.choice(input_data[14:])  # we start only at 15 because the first 14 are handmade samples, where no sibling exist.
        input_data.remove(element_to_pick)

        # find the sibling which shares the same sql
        for sibling in input_data:
            if element_to_pick['query'] == sibling['query']:

                ouput_data_handpicked.append(sibling)
                ouput_data_handpicked.append(element_to_pick)

                input_data.remove(sibling)
                break

    assert len(input_data) == initial_size - n_samples * 2, "Not all samples were picked"

    # set them all to true - we don't want evaluation data which is failing by default
    for handpicked in ouput_data_handpicked:
        handpicked['all_values_found'] = True

    with open(file_output_1, 'w') as f:
        json.dump(input_data, f)

    with open(file_output_2, 'w') as f:
        json.dump(ouput_data_handpicked, f)

    print(f"We handpicked {len(ouput_data_handpicked)} samples and stored them in {file_output_2}. The remaining data is stored in {file_output_1}")


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--file_input', type=str, default='data/cordis/train.json')
    arg_parser.add_argument('--file_output_1', type=str, default='data/cordis/train_reduced.json')
    arg_parser.add_argument('--file_output_2', type=str, default='data/cordis/use_as_evaluation.json')
    arg_parser.add_argument('--n_samples', type=int, default=80, help='Number of samples to pick. '
                                                                      'Be aware that it is multiplied by two, as we always pick the sibling with same SQL, but different NL')

    args = arg_parser.parse_args()

    pick_siblings(args.file_input, args.file_output_1, args.file_output_2, args.n_samples)