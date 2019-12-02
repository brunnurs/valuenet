import random

import numpy as np
import torch


def set_seed_everywhere(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    print("We use the device: '{}' and {} gpu's.".format(device, n_gpu))

    return device, n_gpu


def create_labels_for_dummy_task(data_train, data_dev):
    label_map = {}
    next_label = 0
    for label in data_train:
        if not label['db_id'] in label_map:
            label_map[label['db_id']] = next_label
            next_label += 1

    for label in data_dev:
        if not label['db_id'] in label_map:
            label_map[label['db_id']] = next_label
            next_label += 1

    return label_map


def label_map_values(label_map):
    return list(map(lambda key, value: value, label_map.items()))
