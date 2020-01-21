import torch
from torch import nn
from transformers import BertConfig, BertModel, BertTokenizer

from src.model.encoder.input_features import encode_input


def get_encoder_model(pretrained_model):
    print("load pretrained model/tokenizer for '{}'".format(pretrained_model))
    config_class, model_class, tokenizer_class = (BertConfig, BertModel, BertTokenizer)
    config = config_class.from_pretrained(pretrained_model)
    tokenizer = tokenizer_class.from_pretrained(pretrained_model)
    model = model_class.from_pretrained(pretrained_model, config=config)

    return model, tokenizer


class TransformerEncoder(nn.Module):

    def __init__(self, pretrained_model, device, max_sequence_length):
        super(TransformerEncoder, self).__init__()

        self.max_sequence_length = max_sequence_length
        self.device = device

        config_class, model_class, tokenizer_class = (BertConfig, BertModel, BertTokenizer)

        config = config_class.from_pretrained(pretrained_model)
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_model)
        self.model = model_class.from_pretrained(pretrained_model, config=config)

        # We don't wanna do basic tokenizing (so splitting up a sentence into tokens) as this is already done in pre-processing.
        # But we still wanna do the wordpiece-tokenizing.
        self.tokenizer.do_basic_tokenize = False

    def forward(self, question_tokens, column_names, table_names):
        input_ids_tensor, attention_mask_tensor, segment_ids_tensor, input_lengths = encode_input(question_tokens,
                                                                                                  column_names,
                                                                                                  table_names,
                                                                                                  self.tokenizer,
                                                                                                  self.max_sequence_length,
                                                                                                  self.device)

        outputs = self.model(input_ids_tensor, attention_mask_tensor, segment_ids_tensor)

        last_hidden_states = outputs[0]

        (all_question_span_lengths, all_column_token_lengths, all_table_token_lengths) = input_lengths

        averaged_hidden_states_question = self._average_hidden_states_question(last_hidden_states, all_question_span_lengths)

        # the initial_pointers is basically where the first column starts after the question tokens (including the [CLS] token in the beginning and the [SEP] token in the end)
        initial_pointers =list(map(lambda question_span_lenghts: sum(question_span_lenghts), all_question_span_lengths))

        column_representation = self._build_column_representations_by_rnn(last_hidden_states, all_column_token_lengths, initial_pointers)

        return last_hidden_states

    @staticmethod
    def _average_hidden_states_question(last_hidden_states, all_question_span_lengths):
        """
        As described in the IRNet-paper, we will just average over the sub-tokens of a question-span.
        @param last_hidden_states:
        @param all_question_span_lengths:
        @return:
        """
        all_averaged_hidden_states = []

        for batch_itr_idx, question_span_lengths in enumerate(all_question_span_lengths):
            in_span_pointer = 1  # we start with pointer 1 - remember, the first hidden state is the special [CLS] token, which we don't need.
            averaged_hidden_states = []

            # the first span_length represents the [CLS] token, the last one the [SEP] - we only want the one in between!
            for idx in range(1, len(question_span_lengths) - 1):
                span_length = question_span_lengths[idx]

                averaged_span = torch.mean(last_hidden_states[batch_itr_idx, in_span_pointer: in_span_pointer + span_length, :], keepdim=True, dim=0)
                averaged_hidden_states.append(averaged_span)
                in_span_pointer += span_length

            all_averaged_hidden_states.append(torch.cat(averaged_hidden_states, dim=0))

        return all_averaged_hidden_states

    @staticmethod
    def _build_column_representations_by_rnn(last_hidden_states, all_column_token_lengths, initial_pointers):
        """
        For schema-information (Column and Tables) we do not simply average over the sub-tokens, but run an RNN (Bidirectional LSTM)
        over it and use the last hidden state as representation of the Column/Table
        @param last_hidden_states:
        @param all_column_token_lengths:
        @param initial_pointer:
        """
        all_column_representation = []

        for batch_itr_idx, column_token_lengths, initial_pointer in enumerate(zip(all_column_token_lengths, initial_pointers)):
            for



