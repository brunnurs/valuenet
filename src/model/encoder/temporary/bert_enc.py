import torch
import torch.nn as nn

from transformers import BertTokenizer, BertModel
import collections
from transformers import WordpieceTokenizer
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class Bert_Layer(nn.Module):
    back_type = ['<table>', '<column>', '<agg>', '<MORE>', '<MOST>', '<value>']

    def extend_bert_vocab(self, words_to_extend):
        # print(all_words)

        init_len = len(self.tokenizer.vocab)
        cur_ind = init_len
        for i in words_to_extend:
            if i in self.tokenizer.vocab:
                continue
            self.tokenizer.vocab[i] = cur_ind
            cur_ind += 1

        print(f"extend bert tokenizer with extra {cur_ind - init_len} words!")
        self.tokenizer.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.tokenizer.vocab.items()])
        self.tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.tokenizer.vocab,
                                                                unk_token=self.tokenizer.unk_token)
        self.encoder._resize_token_embeddings(cur_ind)

    def __init__(self, args):
        super(Bert_Layer, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        # self.extend_bert_vocab(self.back_type)
        # for param in self.encoder.embeddings.word_embeddings.parameters():
        #     param.requires_grad = False

        if args.cuda:
            self.new_long_tensor = torch.cuda.LongTensor
            self.new_tensor = torch.cuda.FloatTensor
        else:
            self.new_long_tensor = torch.LongTensor
            self.new_tensor = torch.FloatTensor

        torch.manual_seed(0)
        self.dim_transform = nn.Linear(768, args.hidden_size)
        torch.manual_seed(1)
        self.column_enc = nn.LSTM(768, args.col_embed_size // 2, bidirectional=True,
                                  batch_first=True)
        torch.manual_seed(2)
        self.table_enc = nn.LSTM(768, args.col_embed_size // 2, bidirectional=True,
                                 batch_first=True)

    def forward(self, src_sents, src_sents_len, col_names, table_names):
        '''
        :param src_sents_var: [[span,span,...,span]] * batch_size; span=[word1,word2,...] (word1 might be 'column' 'table' and so on)
        :param src_sents_len: [span_len] * batch_size (descending order)
        :param col_names: [[col,col,...col]] * batch_size; col=[word1,word2,...]
        :return:
        '''
        # print(f"src_sents:\t{src_sents}")
        # print(f"src_sents_len:\t{src_sents_len}")
        # print(f"col_names:\t{col_names}")

        bert_input, infos = self._formatting(src_sents, col_names, table_names)
        # print(f"bert_input:\t{bert_input}")
        padded_input = self._pad_input(bert_input)
        hidden_states, last_cell = self.encoder(padded_input)
        # print(hidden_states.shape)
        # print(infos)

        sent_bert_outs, col_bert_outs, table_bert_outs = [], [], []
        col_split, table_split = [0], [0]
        col_lens, table_lens = [], []
        for batch_iter, formatted_info in enumerate(infos):
            sent_h = []
            span_ptr = 1
            for i in formatted_info['sent']:
                span_h = torch.mean(hidden_states[batch_iter, span_ptr:span_ptr + i, :], dim=0, keepdim=True)
                sent_h.append(span_h)
                span_ptr += i
            sent_bert_outs.append(torch.cat(sent_h, dim=0))

            col_split.append(col_split[-1] + len(formatted_info['col']))
            col_lens += formatted_info['col']
            for i in formatted_info['col']:
                span_ptr += 1
                col_bert_outs.append(hidden_states[batch_iter, span_ptr:span_ptr + i, :])
                span_ptr += i

            table_split.append(table_split[-1] + len(formatted_info['table']))
            table_lens += formatted_info['table']
            for i in formatted_info['table']:
                span_ptr += 1
                table_bert_outs.append(hidden_states[batch_iter, span_ptr:span_ptr + i, :])
                span_ptr += i

        assert src_sents_len == [x.shape[0] for x in
                                 sent_bert_outs], f'{src_sents_len} vs {[x.shape[0] for x in sent_bert_outs]}'
        sent_outs = pad_sequence(sent_bert_outs, batch_first=True)  # bsize * max_sents_len * 768
        sent_outs = self.dim_transform(sent_outs)
        print("dummy2")
        print(sent_outs[0])

        col_bert_outs = pad_sequence(col_bert_outs, batch_first=True)  # total_col_num * max_col_len * 768
        table_bert_outs = pad_sequence(table_bert_outs, batch_first=True)  # total_table_num * max_table_len * 768
        col_lens = self.new_long_tensor(col_lens)
        table_lens = self.new_long_tensor(table_lens)

        _, (col_last_states, _) = rnn_wrapper(self.column_enc, col_bert_outs, col_lens)
        col_lstm_outs = torch.cat([col_last_states[0], col_last_states[1]], -1)  # total_col_num * hidden_size
        assert col_lstm_outs.shape[0] == col_split[-1]
        col_outs = [col_lstm_outs[col_split[i]:col_split[i + 1]] for i in range(len(col_split) - 1)]
        col_outs = pad_sequence(col_outs, batch_first=True)

        _, (table_last_states, _) = rnn_wrapper(self.table_enc, table_bert_outs, table_lens)
        table_lstm_outs = torch.cat([table_last_states[0], table_last_states[1]], -1)
        assert table_lstm_outs.shape[0] == table_split[-1]
        table_outs = [table_lstm_outs[table_split[i]:table_split[i + 1]] for i in range(len(table_split) - 1)]
        table_outs = pad_sequence(table_outs, batch_first=True)

        # print(f'sent_bert_outs:\t{sent_bert_outs.shape}')
        # print(f'col_bert_outs:\t{col_bert_outs.shape}')
        # print(f'table_bert_outs:\t{table_bert_outs.shape}')
        # print(sent_bert_outs)

        return sent_outs, col_outs, table_outs, last_cell

    def _formatting(self, src_sents, col_names, table_names):
        formatted_inputs = []
        formatted_infos = []
        for src_sent, col_name, table_name in zip(src_sents, col_names, table_names):
            formatted_info = {"sent": [], "col": [], "table": []}
            formatted = [self.tokenizer.cls_token]
            for span in src_sent:
                formatted_info['sent'].append(len(span))
                formatted += span
            formatted += [self.tokenizer.sep_token]
            for i in col_name:
                formatted_info['col'].append(len(i))
                formatted += i
                formatted += [self.tokenizer.sep_token]
            for i in table_name:
                formatted_info['table'].append(len(i))
                formatted += i
                formatted += [self.tokenizer.sep_token]
            formatted_inputs.append(formatted)
            formatted_infos.append(formatted_info)
        return formatted_inputs, formatted_infos

    def _pad_input(self, bert_inp):
        bert_lens = [len(i) for i in bert_inp]
        max_len = max(bert_lens)
        padded_inp = [p + [self.tokenizer.pad_token] * (max_len - i) for i, p in zip(bert_lens, bert_inp)]

        padded_inp = self.new_long_tensor([self.tokenizer.convert_tokens_to_ids(i) for i in padded_inp])

        return padded_inp


def rnn_wrapper(encoder, inputs, lens, cell='lstm'):
    """
        @args:
            encoder(nn.Module): rnn series bidirectional encoder, batch_first=True
            inputs(torch.FloatTensor): rnn inputs, bsize x max_seq_len x in_dim
            lens(torch.LongTensor): seq len for each sample, bsize
        @return:
            out(torch.FloatTensor): output of encoder, bsize x max_seq_len x hidden_dim*2
            hidden_states(tuple or torch.FloatTensor): final hidden states, num_layers*2 x bsize x hidden_dim
    """
    # rerank according to lens and temporarily remove empty inputs
    sorted_lens, sort_key = torch.sort(lens, descending=True)
    nonzero_index = torch.sum(sorted_lens > 0).item()
    sorted_inputs = torch.index_select(inputs, dim=0, index=sort_key[:nonzero_index])
    # forward non empty inputs
    packed_inputs = pack_padded_sequence(sorted_inputs, sorted_lens[:nonzero_index].tolist(), batch_first=True)
    packed_out, h = encoder(packed_inputs)  # bsize x srclen x dim
    out, _ = pad_packed_sequence(packed_out, batch_first=True)
    if cell.upper() == 'LSTM':
        h, c = h
    # pad zeros due to empty inputs
    pad_zeros = torch.zeros(sorted_lens.size(0) - out.size(0), out.size(1), out.size(2)).type_as(out).to(out.device)
    sorted_out = torch.cat([out, pad_zeros], dim=0)
    pad_hiddens = torch.zeros(h.size(0), sorted_lens.size(0) - h.size(1), h.size(2)).type_as(h).to(h.device)
    sorted_hiddens = torch.cat([h, pad_hiddens], dim=1)
    if cell.upper() == 'LSTM':
        pad_cells = torch.zeros(c.size(0), sorted_lens.size(0) - c.size(1), c.size(2)).type_as(c).to(c.device)
        sorted_cells = torch.cat([c, pad_cells], dim=1)
    # rerank according to sort_key
    shape = list(sorted_out.size())
    out = torch.zeros_like(sorted_out).type_as(sorted_out).to(sorted_out.device).scatter_(0, sort_key.unsqueeze(
        -1).unsqueeze(-1).expand(*shape), sorted_out)
    shape = list(sorted_hiddens.size())
    hiddens = torch.zeros_like(sorted_hiddens).type_as(sorted_hiddens).to(sorted_hiddens.device).scatter_(1,
                                                                                                          sort_key.unsqueeze(
                                                                                                              0).unsqueeze(
                                                                                                              -1).expand(
                                                                                                              *shape),
                                                                                                          sorted_hiddens)
    if cell.upper() == 'LSTM':
        cells = torch.zeros_like(sorted_cells).type_as(sorted_cells).to(sorted_cells.device).scatter_(1,
                                                                                                      sort_key.unsqueeze(
                                                                                                          0).unsqueeze(
                                                                                                          -1).expand(
                                                                                                          *shape),
                                                                                                      sorted_cells)
        return out, (hiddens.contiguous(), cells.contiguous())
    return out, hiddens.contiguous()
