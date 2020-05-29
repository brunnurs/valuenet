import codecs
import datetime
import os

import random

import numpy as np
import torch
import wandb


def create_experiment_folder(model_output_dir, name):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    exp = "{}__{}".format(name, timestamp)

    out_path = os.path.join(model_output_dir, exp)
    os.makedirs(out_path, exist_ok=True)

    return exp, out_path


def set_seed_everywhere(seed, n_gpu):
    random.seed(int(seed * 13 / 7))
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
    with open(file_name, encoding='utf-8') as inf:
        for idx, line in enumerate(inf):
            if (use_small and idx >= 500000):
                break
            info = line.strip().split(' ')
            if info[0].lower() not in ret:
                ret[info[0]] = np.array(list(map(lambda x: float(x), info[1:])))
    return ret


def load_word_emb_binary(embedding_file_name_w_o_suffix):
    print("Loading binary word embedding from {0}.vocab and {0}.npy".format(embedding_file_name_w_o_suffix))

    with codecs.open(embedding_file_name_w_o_suffix + '.vocab', 'r', 'utf-8') as f_in:
        index2word = [line.strip() for line in f_in]

    wv = np.load(embedding_file_name_w_o_suffix + '.npy')
    word_embedding_map = {}
    for i, w in enumerate(index2word):
        word_embedding_map[w] = wv[i]

    return word_embedding_map


def save_model(model, model_save_path, model_name="best_model.pt"):
    torch.save(model.state_dict(), os.path.join(model_save_path, model_name))
    # also save the model to "Weights & Biases"
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'best_model.pt'))
