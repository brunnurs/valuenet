import argparse
import codecs

import numpy as np
from pytictoc import TicToc

from utils import load_word_emb_binary, load_word_emb


def convert_to_binary(embedding_path):
    f = codecs.open(embedding_path + ".txt", 'r', encoding='utf-8')
    wv = []
    with codecs.open(embedding_path + ".vocab", "w", encoding='utf-8') as vocab_write:
        count = 0
        for line in f:
            splitlines = line.split()
            vocab_write.write(splitlines[0].strip())
            vocab_write.write("\n")
            wv.append([float(val) for val in splitlines[1:]])
        count += 1

    np.save(embedding_path + ".npy", np.array(wv))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parse word embeddings in text format.")

    # evaluation
    parser.add_argument('--embedding_text_file', help="Specify the path without file-suffix!", required=True, type=str)

    args = parser.parse_args()

    print("Convert embedding file '{0}.txt' to the binary file '{0}.npy' and the vocab-file '{0}.vocab'".format(args.embedding_text_file))
    convert_to_binary(args.embedding_text_file)

    print("Converting done! Try to reload.")
    t = TicToc()
    t.tic()
    word_embedding_map = load_word_emb_binary(args.embedding_text_file)
    loading_time = t.tocvalue()
    print("Load {} words and embeddings in {} seconds".format(len(word_embedding_map), loading_time))

    t.tic()
    word_embedding_map_2 = load_word_emb(args.embedding_text_file + '.txt')
    t.toc(msg="Loading it as text file takes")
