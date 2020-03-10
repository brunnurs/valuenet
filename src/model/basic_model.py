import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from intermediate_representation.semQL import N, A, C, T, V, Filter, Order, Sup


class BasicModel(nn.Module):

    def __init__(self):
        super(BasicModel, self).__init__()
        pass

    def embedding_cosine(self, src_embedding, table_embedding, table_unk_mask):
        embedding_differ = []
        for i in range(table_embedding.size(1)):
            one_table_embedding = table_embedding[:, i, :]
            one_table_embedding = one_table_embedding.unsqueeze(1).expand(table_embedding.size(0),
                                                                          src_embedding.size(1),
                                                                          table_embedding.size(2))

            topk_val = F.cosine_similarity(one_table_embedding, src_embedding, dim=-1)

            embedding_differ.append(topk_val)
        embedding_differ = torch.stack(embedding_differ).transpose(1, 0)
        embedding_differ.data.masked_fill_(table_unk_mask.unsqueeze(2).expand(
            table_embedding.size(0),
            table_embedding.size(1),
            embedding_differ.size(2)
        ).bool(), 0)

        return embedding_differ

    def encode(self, src_sents_var, src_sents_len, q_onehot_project=None):
        """
        encode the source sequence
        :return:
            src_encodings: Variable(batch_size, src_sent_len, hidden_size * 2)
            last_state, last_cell: Variable(batch_size, hidden_size)
        """
        # Look up word embeddings for both, src words and types (e.g. "column", "table", etc.). After this we have the
        # matrix: 64 (batchsize) * 22 (input tokens --> only the question words, not the columns) * 300 (embedding size)
        src_token_embed = self.gen_x_batch(src_sents_var)

        if q_onehot_project is not None:
            src_token_embed = torch.cat([src_token_embed, q_onehot_project], dim=-1)
        # A pytorch util to easily fill a batch with sequences of different size.
        packed_src_token_embed = pack_padded_sequence(src_token_embed, src_sents_len, batch_first=True)
        # src_encodings: (tgt_query_len, batch_size, hidden_size)

        # This is actually the NL-Encoder as in the paper: A multi-layer LSTM wich creates the source-embeddings H_x.
        # Important: the output is one value per time step t, as well as the hidden state/cell state of the last cell.
        src_encodings, (last_state, last_cell) = self.encoder_lstm(packed_src_token_embed)
        # the inverse operation for pack_padded_sequence above.
        src_encodings, _ = pad_packed_sequence(src_encodings, batch_first=True)
        # src_encodings: (batch_size, tgt_query_len, hidden_size)
        # src_encodings = src_encodings.permute(1, 0, 2)
        # (batch_size, hidden_size * 2)

        # This concatenation is because we have two cell state (it's a bi-directional LSTM). Both, last_state[0,1] have a size of 150 before the concatenation.
        last_state = torch.cat([last_state[0], last_state[1]], -1)
        last_cell = torch.cat([last_cell[0], last_cell[1]], -1)

        return src_encodings, (last_state, last_cell)

    def input_type(self, values_list):
        """
        Notes Ursin: this is only creating a pytorch tensor from the "col_hot_type" array. Nothing fancy here.
        @param values_list:
        @return:
        """
        B = len(values_list)
        val_len = []
        for value in values_list:
            val_len.append(len(value))
        max_len = max(val_len)
        # for the Begin and End
        val_emb_array = np.zeros((B, max_len, values_list[0].shape[1]), dtype=np.float32)
        for i in range(B):
            val_emb_array[i, :val_len[i], :] = values_list[i][:, :]

        val_inp = torch.from_numpy(val_emb_array)
        if self.args.cuda:
            val_inp = val_inp.cuda()
        val_inp_var = Variable(val_inp)
        return val_inp_var

    def padding_sketch(self, sketch):
        """
        Padding the sketch with leaf actions (A, C and T) where necessary.
        While we still don't know the id_c of the leaf actions, we know based on the grammar exactly, where to insert one.
        @param sketch:
        @return:
        """
        padding_result = []
        for action in sketch:
            padding_result.append(action)
            if type(action) == N:
                for _ in range(action.id_c + 1):
                    padding_result.append(A(0))
                    padding_result.append(C(0))
                    padding_result.append(T(0))
            elif type(action) == Filter:
                padding_result.extend(self._padd_filter(action))
            elif type(action) == Order or type(action) == Sup:
                padding_result.append(A(0))
                padding_result.append(C(0))
                padding_result.append(T(0))

        return padding_result

    @staticmethod
    def _padd_filter(action):
        if 'A' in action.production:
            filter_paddings = []
            start_idx = action.production.index('A')
            all_padding_objects = action.production[start_idx:].split(' ')

            for e in all_padding_objects:
                if e == 'A':
                    filter_paddings.append(A(0))
                    filter_paddings.append(C(0))
                    filter_paddings.append(T(0))
                elif e == 'V':
                    filter_paddings.append(V(0))
                elif e == 'Root':
                    # we don't need to do anything for 'Root' -> it will be padded later.
                    continue
                else:
                    raise ValueError("Unknown Action: " + e)

            return filter_paddings
        else:
            return []

    def gen_x_batch(self, q):
        B = len(q)
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        is_list = False
        if type(q[0][0]) == list:
            is_list = True
        for i, one_q in enumerate(q):
            if not is_list:
                q_val = list(
                    map(lambda x: self.word_emb.get(x, np.zeros(self.args.col_embed_size, dtype=np.float32)), one_q))
            else:
                q_val = []
                for ws in one_q:
                    emb_list = []
                    ws_len = len(ws)
                    for w in ws:
                        emb_list.append(self.word_emb.get(w, self.word_emb['unk']))
                    if ws_len == 0:
                        raise Exception("word list should not be empty!")
                    elif ws_len == 1:
                        q_val.append(emb_list[0])
                    else:
                        # there can be multiple tokens if the schema linking already found a type ("column", "table", "value") for a span (word). In that case,
                        # we concat the embeddings and normalize them.
                        q_val.append(sum(emb_list) / float(ws_len))

            val_embs.append(q_val)
            val_len[i] = len(q_val)
        max_len = max(val_len)

        val_emb_array = np.zeros((B, max_len, self.args.col_embed_size), dtype=np.float32)
        for i in range(B):
            for t in range(len(val_embs[i])):
                val_emb_array[i, t, :] = val_embs[i][t]
        val_inp = torch.from_numpy(val_emb_array)
        if self.args.cuda:
            val_inp = val_inp.cuda()
        return val_inp

    def save(self, path):
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        params = {
            'args': self.args,
            'vocab': self.vocab,
            'grammar': self.grammar,
            'state_dict': self.state_dict()
        }
        torch.save(params, path)
