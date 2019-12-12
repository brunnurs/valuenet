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


def load_word_emb(file_name, use_small=False):
    print('Loading word embedding from %s' % file_name)
    ret = {}
    with open(file_name) as inf:
        for idx, line in enumerate(inf):
            if (use_small and idx >= 500000):
                break
            info = line.strip().split(' ')
            if info[0].lower() not in ret:
                ret[info[0]] = np.array(list(map(lambda x: float(x), info[1:])))
    return ret
