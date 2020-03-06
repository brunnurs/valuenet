import numbers

import nltk
import torch
from word2number import w2n


################## NOTE URSIN #############################
# This code is currently not used, but an idea which might need to be followed up at a later point.
###########################################################

def create_value_encodings_from_question_tokens(all_last_hidden_states, all_question_span_lengths, all_question_tokens, all_values, device):
    """
    Create encodings for all values in the question. Values can be strings (e.g. "USA"), numbers (e.g. 4.5) or more
    specific formats (e.g. "%partialvalue% if use the fuzzy SQL "LIKE" logic). This method is trying to find given values in the question tokens
    and then averages over all token of a value with an RNN (or with simple averaging).
    Where the values come from (ground truth, NER, using the data of the database, etc.) is not part of this method but need to be done in a pre-processing step.
    @param all_last_hidden_states: the hidden states after transformer encoding. Be aware that one question token can be tokenized in wordpieces.
    @param all_question_span_lengths:
    @param all_question_tokens: The question, tokenized.
    @param all_values: The values we try to encode. Be aware that the values need to be detected in a pre-processing step.
    @param device: necessary in case we create a new, empty tensor.
    @return:
    """
    all_value_hidden_states = []
    error_count = 0
    for batch_itr_idx, (question_tokens, hidden_states, span_lengths, values) in enumerate(zip(all_question_tokens, all_last_hidden_states, all_question_span_lengths, all_values)):
        value_hidden_states = _create_value_encodings(hidden_states, question_tokens, span_lengths, values, device)

        # assert len(values) == value_hidden_states.shape[0], "some values could not get found in the question!"
        all_value_hidden_states.append(value_hidden_states)

    return all_value_hidden_states


def _create_value_encodings(hidden_states, question_tokens, question_span_lengths, values, device):
    avg_value_hidden_states = []

    # make sure we keep the order of the values, as this is the order we use in the ground truth (e.g. V(5)).
    for value in values:
        # as a value can include multiple tokens, we first need to find all of them consecutive in the question
        relevant_token_indices = _find_value_in_question_tokens(question_tokens, value)

        if not relevant_token_indices:
            print('{}   {}'.format(value, question_tokens))
            continue

        relevant_hidden_states = _select_relevant_hidden_states(hidden_states, question_span_lengths, relevant_token_indices)

        # For each value we average over all subtokens. We use torch.stack() to create from the list of tensors a new tensor with dimensions (n, 768).
        # keep in mind that torch.cat() would not work here, as we would create one big tensor with dimensions (n*768)
        avg_value_hidden_states.append(torch.mean(torch.stack(relevant_hidden_states), dim=0, keepdim=True))

    if avg_value_hidden_states:
        # here we use torch.cat(), as the list already hast the proper dimensions [(1, 768), (1, 768), (1, 768)...] and  we just wanna create one big tensor with dimensions (n, 768)
        return torch.cat(avg_value_hidden_states, dim=0)
    else:
        # what if there is no values for that record? Create an empty tensor with correct dimensions, so we can padd those rows correctly.
        return torch.empty(0, hidden_states.shape[1]).to(device)


def _find_value_in_question_tokens(question_tokens, value):

    # there is some rare cases where the value is another column. We ignore them for now
    if isinstance(value, list):
        return None

    # there is a few cases where the value is an empty string, but actually it refers to the word "empty" in the sentece.
    if isinstance(value, str) and not value:
        value = "empty"

    if not isinstance(value, numbers.Number):
        value_tokenized = _tokenize(value)
    else:
        value_tokenized = [value]

    # we start by finding the first token of the value in the question - as soon as we found it, we start looking for all expected subsequent tokens
    first_value_token = value_tokenized[0]

    for idx, token in enumerate(question_tokens):
        # a token can again have multiple sub-tokens, due to the subword-tokenizing and adding the value-type
        if _is_any_sub_token_a_value(token, first_value_token):
            # we found the first token - now we expect all subsequent tokens of the value to follow.
            all_subtokens_found = True
            window_idx = idx + 1
            for value_token in value_tokenized[1:]:
                # with this if we make sure our sliding window does not exceed the end of question_tokens. This could be the case if we hit a match by the end of the string.
                if window_idx < len(question_tokens):
                    token_in_window = question_tokens[window_idx]
                    if not _is_any_sub_token_a_value(token_in_window, value_token):
                        all_subtokens_found = False
                    window_idx += 1
                else:
                    break

            if all_subtokens_found:
                return range(idx, window_idx)

    return None


def _select_relevant_hidden_states(hidden_states, question_span_lengths, relevant_token_indices):
    relevant_hidden_states = []
    for token_idx in relevant_token_indices:
        # there is one hidden state for each wordpiece. To get the start-idx of the wordpieces we currently try to select,
        # we therefore need to sum up all questions spans before the current one.
        hidden_states_idx = sum(question_span_lengths[:token_idx])

        number_of_tokens_to_select = question_span_lengths[token_idx]

        relevant_hidden_states.extend(hidden_states[hidden_states_idx:hidden_states_idx + number_of_tokens_to_select])

    return relevant_hidden_states


def _is_any_sub_token_a_value(token, value):

    if isinstance(value, str):
        if value.startswith('%'):
            value = value[1:]

        if value.endswith('%'):
            value = value[:-1]

    for sub_token in token:
        if isinstance(value, numbers.Number):

            # there are cases where we have an int in the string (e.g. '300') and a float as value from the ground truth (e.g. 300.0)
            # to match such cases, we convert back to numbers.
            if _is_int(sub_token) and isinstance(value, float):
                value_int = int(value)
                subtoken_int = int(sub_token)
                if value_int == subtoken_int:
                    return True

            if _is_int(sub_token) and isinstance(value, int):
                subtoken_int = int(sub_token)
                if value == subtoken_int:
                    return True

            token_to_number = _word_to_number(sub_token)
            if token_to_number == value:
                return True

        else:
            # often value tokens start/end with an apostrophe (be aware that there are multiple types of apostrophes).
            # As this value is not available in the values (and would also not be in a real world question) we remove it.
            if sub_token.startswith(('\'', '‘', '“')) and len(sub_token) > 2:
                sub_token = sub_token[1:]

            if sub_token.endswith(('\'', '’', '”')) and len(sub_token) > 1:
                sub_token = sub_token[:-1]

        # after handling specific numeric cases, we don't care anymore if the value is an int or a string (be aware from a TTS interface we anyway don't get this information)
        # to compare the value with the token from the question, we need to treat both as strings.
        value_str = str(value)

        value_str = value_str.lower()
        sub_token = sub_token.lower()

        if value_str == sub_token:
            return True

    return False


def _tokenize(value):
    if not (value.startswith('%') or value.endswith('%')):
        return nltk.word_tokenize(value)
    else:
        return [value]


def _is_int(s):
    if s[0] in ('-', '+'):
        return s[1:].isdigit()
    return s.isdigit()


def _word_to_number(word):
    try:
        number = w2n.word_to_num(word)
        return number
    except ValueError:
        return None

