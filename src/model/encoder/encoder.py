import torch
from more_itertools import flatten
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertConfig, BertModel, BertTokenizer

from model.encoder.input_features import encode_input


def get_encoder_model(pretrained_model):
    print("load pretrained model/tokenizer for '{}'".format(pretrained_model))
    config_class, model_class, tokenizer_class = (BertConfig, BertModel, BertTokenizer)
    config = config_class.from_pretrained(pretrained_model)
    tokenizer = tokenizer_class.from_pretrained(pretrained_model)
    model = model_class.from_pretrained(pretrained_model, config=config)

    return model, tokenizer


class TransformerEncoder(nn.Module):

    def __init__(self, pretrained_model, device, max_sequence_length, schema_embedding_size, decoder_hidden_size):
        super(TransformerEncoder, self).__init__()

        self.max_sequence_length = max_sequence_length
        self.device = device

        config_class, model_class, tokenizer_class = (BertConfig, BertModel, BertTokenizer)

        transformer_config = config_class.from_pretrained(pretrained_model)
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_model)
        self.transformer_model = model_class.from_pretrained(pretrained_model, config=transformer_config)

        self.encoder_hidden_size = transformer_config.hidden_size

        # We don't wanna do basic tokenizing (so splitting up a sentence into tokens) as this is already done in pre-processing.
        # But we still wanna do the wordpiece-tokenizing.
        self.tokenizer.do_basic_tokenize = False

        self.linear_layer_dimension_reduction_question = nn.Linear(transformer_config.hidden_size, decoder_hidden_size)

        self.column_encoder = nn.LSTM(transformer_config.hidden_size, schema_embedding_size // 2, bidirectional=True, batch_first=True)
        self.table_encoder = nn.LSTM(transformer_config.hidden_size, schema_embedding_size // 2, bidirectional=True, batch_first=True)
        self.value_encoder = nn.LSTM(transformer_config.hidden_size, schema_embedding_size // 2, bidirectional=True, batch_first=True)

        print("Successfully loaded pre-trained transformer '{}'".format(pretrained_model))

    def forward(self, question_tokens, column_names, table_names, values):
        input_ids_tensor, attention_mask_tensor, segment_ids_tensor, input_lengths = encode_input(question_tokens,
                                                                                                  column_names,
                                                                                                  table_names,
                                                                                                  values,
                                                                                                  self.tokenizer,
                                                                                                  self.max_sequence_length,
                                                                                                  self.device)

        # while the "last_hidden-states" is one hidden state per input token, the pooler_output is the hidden state of the [CLS]-token, further processed.
        # See e.g. "BertModel" documentation for more information.

        last_hidden_states, pooling_output = self.transformer_model(input_ids_tensor, attention_mask_tensor, segment_ids_tensor)

        (all_question_span_lengths, all_column_token_lengths, all_table_token_lengths, all_value_token_lengths) = input_lengths

        # we get the relevant hidden states for the question-tokens and average, if there are multiple token per  word (e.g ['table', 'college'])
        averaged_hidden_states_question, pointers_after_question = self._average_hidden_states_question(last_hidden_states, all_question_span_lengths)
        question_out = pad_sequence(averaged_hidden_states_question, batch_first=True)  # (batch_size * max_question_tokens_per_batch * hidden_size)
        # as the transformer uses normally a size of 768 and the decoder only 300 per vector, we need to reduce dimensionality here with a linear layer.
        question_out = self.linear_layer_dimension_reduction_question(question_out)

        column_hidden_states, pointers_after_columns = self._get_schema_hidden_states(last_hidden_states, all_column_token_lengths, pointers_after_question)
        table_hidden_states, pointers_after_tables = self._get_schema_hidden_states(last_hidden_states, all_table_token_lengths, pointers_after_columns)

        # in this scenario, we know the values upfront and encode them similar to tables/columns. This is different as soon as we don't know the values but have to extract them from the question.
        value_hidden_states, pointers_after_values = self._get_schema_hidden_states(last_hidden_states, all_value_token_lengths, pointers_after_tables)

        # This is simply to make sure the rather complex token-concatenation happens correctly. Can get removed at some point.
        self._assert_all_elements_processed(all_question_span_lengths,
                                            all_column_token_lengths,
                                            all_table_token_lengths,
                                            all_value_token_lengths,
                                            pointers_after_values,
                                            last_hidden_states.shape[1])

        # "column_hidden_states" (and table_hidden_states/value_hidden_states) is here a list of examples, with each example a list of tensors (one tensor for each column). As a column can have multiple words, the tensor consists of multiple columns (e.g. 3 * 768)
        # With this line we first concat all examples to one huge list of tensors, independent of the example. Remember: we don't wanna use an RNN over a full example - but only over the tokens of ONE column! Therefore we can just build up a batch of each column - tensor.
        # With "pad_sequence" we pay attention to the fact that each column can have a different amount of tokens (e.g. a 3-word column vs. a 1 word column), so we have to pad the shorter inputs.
        column_hidden_states_padded = pad_sequence(list(flatten(column_hidden_states)), batch_first=True)
        column_lengths = [len(t) for t in flatten(column_hidden_states)]

        table_hidden_states_padded = pad_sequence(list(flatten(table_hidden_states)), batch_first=True)
        table_lengths = [len(t) for t in flatten(table_hidden_states)]

        # create one embedding for each column by using an RNN.
        _, column_last_states, _ = self._rnn_wrapper(self.column_encoder, column_hidden_states_padded, column_lengths)

        # create one embedding for each table by using an RNN.
        _, table_last_states, _ = self._rnn_wrapper(self.table_encoder, table_hidden_states_padded, table_lengths)

        assert column_last_states.shape[0] == sum(map(lambda l: len(l), column_hidden_states))
        assert table_last_states.shape[0] == sum(map(lambda l: len(l), table_hidden_states))

        column_out = self._back_to_original_size(column_last_states, column_hidden_states)
        column_out_padded = pad_sequence(column_out, batch_first=True)

        table_out = self._back_to_original_size(table_last_states, table_hidden_states)
        table_out_padded = pad_sequence(table_out, batch_first=True)

        # in contrary to columns/tables there can be no values in a batch. In that case, return an empty tensor.
        if list(flatten(value_hidden_states)):
            value_hidden_states_padded = pad_sequence(list(flatten(value_hidden_states)), batch_first=True)
            value_lengths = [len(t) for t in flatten(value_hidden_states)]

            # create one embedding for each value by using an RNN.
            _, value_last_states, _ = self._rnn_wrapper(self.value_encoder, value_hidden_states_padded, value_lengths)

            assert value_last_states.shape[0] == sum(map(lambda l: len(l), value_hidden_states))

            value_out = self._back_to_original_size(value_last_states, value_hidden_states)
            value_out_padded = pad_sequence(value_out, batch_first=True)
        else:
            value_out_padded = torch.zeros(table_out_padded.shape[0], 0, table_out_padded.shape[2]).to(self.device)

        return question_out, column_out_padded, table_out_padded, value_out_padded, pooling_output

    @staticmethod
    def _average_hidden_states_question(last_hidden_states, all_question_span_lengths):
        """
        As described in the IRNet-paper, we will just average over the sub-tokens of a question-span.
        """
        all_averaged_hidden_states = []
        last_pointers = []

        for batch_itr_idx, question_span_lengths in enumerate(all_question_span_lengths):
            pointer = 1  # we start with pointer 1 - remember, the first hidden state is the special [CLS] token, which we don't need.
            averaged_hidden_states = []

            # the first span_length represents the [CLS] token, the last one the [SEP] - we only want the one in between!
            for idx in range(1, len(question_span_lengths) - 1):
                span_length = question_span_lengths[idx]

                averaged_span = torch.mean(last_hidden_states[batch_itr_idx, pointer: pointer + span_length, :],
                                           keepdim=True, dim=0)
                averaged_hidden_states.append(averaged_span)
                pointer += span_length

            all_averaged_hidden_states.append(torch.cat(averaged_hidden_states, dim=0))
            last_pointers.append(pointer + 1)  # the + 1 represents the [SEP] token in the end

        return all_averaged_hidden_states, last_pointers

    @staticmethod
    def _get_schema_hidden_states(last_hidden_states, all_schema_token_lengths, initial_pointers):
        """
        We simply put together the tokens for each column/table and filter out the separators. No averaging or concatenation, as we will use an RNN later
        """
        all_schema_hidden_state = []
        last_pointers = []

        for batch_itr_idx, (schema_token_lengths, initial_pointer) in enumerate(zip(all_schema_token_lengths, initial_pointers)):
            hidden_states_schema = []
            pointer = initial_pointer
            for schema_token_length in schema_token_lengths:
                # the -1 represents the [SEP] by the end of the column, which we don't wanna include.
                hidden_states_schema.append(last_hidden_states[batch_itr_idx, pointer: pointer + schema_token_length - 1, :])
                pointer += schema_token_length

            all_schema_hidden_state.append(hidden_states_schema)
            last_pointers.append(pointer)

        return all_schema_hidden_state, last_pointers

    def _rnn_wrapper(self, encoder, inputs, lengths):
        """
        This function abstracts from the technical details of the RNN. It handles the whole packing/unpacking of the values,
        handling zero-values and concatenating hidden/cell-states.
        """
        lengths = torch.tensor(lengths).to(self.device)

        # we need to sort the inputs by length due to the use of "pack_padded_sequence" which expects a sorted input.
        sorted_lens, sort_key = torch.sort(lengths, descending=True)
        # we remove temporally remove empty inputs
        nonzero_index = torch.sum(sorted_lens > 0).item()
        sorted_inputs = torch.index_select(inputs, dim=0, index=sort_key[:nonzero_index])

        # Even though we already padded inputs before "_rnn_wrapper", we still  need to "pack/unpack" the sequences.
        # Reason is mostly performance wise, read here: https://stackoverflow.com/a/55805785/1081551
        packed_inputs = pack_padded_sequence(sorted_inputs, sorted_lens[:nonzero_index].tolist(), batch_first=True)

        # forward it to the encoder network
        packed_out, (h, c) = encoder(packed_inputs)
        # unpack afterwards
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # output dimensions:
        # out: (batch_size * max_sequence_length [padded] * dim). Example: (20 * 3 * 768)
        # h: (uni/bi-directional LSTM * batch_size * hidden_size). Example: (2 * 20 * 150). So 2 values per sequence, and the 2 is because we use bi-directional LSTM's
        # c: (uni/bi-directional LSTM * batch_size * hidden_size). Example: (2 * 20 * 150). So 2 values per sequence, and the 2 is because we use bi-directional LSTM's

        # as we remove zero-length inputs before, we need to extend the results here by the zero inputs
        # we do it for output
        pad_zeros = torch.zeros(sorted_lens.size(0) - out.size(0), out.size(1), out.size(2)).type_as(out).to(out.device)
        sorted_out = torch.cat([out, pad_zeros], dim=0)

        # and hidden state
        pad_hiddens = torch.zeros(h.size(0), sorted_lens.size(0) - h.size(1), h.size(2)).type_as(h).to(h.device)
        sorted_hiddens = torch.cat([h, pad_hiddens], dim=1)

        # and cell state
        pad_cells = torch.zeros(c.size(0), sorted_lens.size(0) - c.size(1), c.size(2)).type_as(c).to(c.device)
        sorted_cells = torch.cat([c, pad_cells], dim=1)

        # remember that sorted above and ranked by length? Here we need to invert this sorting to return in the same
        # order as the input was!
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

        cells = torch.zeros_like(sorted_cells).type_as(sorted_cells).to(sorted_cells.device).scatter_(1,
                                                                                                      sort_key.unsqueeze(
                                                                                                          0).unsqueeze(
                                                                                                          -1).expand(
                                                                                                          *shape),
                                                                                                      sorted_cells)

        # contiguous/non-contiguous seems to be a memory implementation detail to me...
        # look at this for more details: https://discuss.pytorch.org/t/contigious-vs-non-contigious-tensor/30107/2
        hiddens = hiddens.contiguous()
        cells = cells.contiguous()

        # here we concat the two hidden states/cell states of the Bi-directional LSTM
        hiddens_concated = torch.cat([hiddens[0], hiddens[1]], -1)
        cells_concated = torch.cat([cells[0], cells[1]], -1)

        return out, hiddens_concated, cells_concated

    @staticmethod
    def _back_to_original_size(elements_to_split, original_array):
        original_split = []

        dimensions = map(lambda l: len(l), original_array)

        current_idx = 0
        for length in dimensions:
            original_split.append(elements_to_split[current_idx:current_idx + length])
            current_idx += length

        assert elements_to_split.shape[0] == current_idx

        return original_split

    @staticmethod
    def _assert_all_elements_processed(all_question_span_lengths, all_column_token_lengths, all_table_token_lengths, all_value_token_lengths, last_pointers, len_last_hidden_states):

        # the longest element in the batch will decide how large the sequence is - therefore the max. pointer is the size of the hidden states.
        assert max(last_pointers) == len_last_hidden_states

        for question_span_lengths, column_token_lengths, table_token_lengths, value_token_length, last_pointer in zip(all_question_span_lengths, all_column_token_lengths, all_table_token_lengths, all_value_token_lengths, last_pointers):
            assert sum(question_span_lengths) + sum(column_token_lengths) + sum(table_token_lengths) + sum(value_token_length) == last_pointer
