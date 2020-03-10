from unittest import TestCase

from transformers import BertTokenizer

from model.encoder.input_features import encode_input, _tokenize_values


class Test(TestCase):
    def test_encode_input(self):
        # GIVEN
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        question_spans = [[['how'], ['many'], ['table', 'student'], ['are'], ['attending'], ['english'], ['french'], ['table', 'course'], ['?']]]
        column_names = [[['count', 'number', 'many'], ['address', 'id'], ['line', '1'], ['line', '2'], ['city'], ['zip', 'postcode'], ['state', 'province', 'county'], ['country'], ['person', 'id'], ['first', 'name'], ['middle', 'name'], ['last', 'name'], ['cell', 'mobile', 'number'], ['email', 'address'], ['login', 'name'], ['password'], ['student', 'id'], ['student', 'detail'], ['course', 'id'], ['course', 'name'], ['course', 'description'], ['other', 'detail'], ['person', 'address', 'id'], ['date', 'from'], ['date', 'to'], ['registration', 'date'], ['date', 'of', 'attendance'], ['candidate', 'id'], ['candidate', 'detail'], ['qualification'], ['assessment', 'date'], ['asessment', 'outcome', 'code']]]
        table_names = [[['address'], ['people'], ['student'], ['course'], ['people', 'address'], ['student', 'course', 'registration'], ['student', 'course', 'attendance'], ['candidate'], ['candidate', 'assessment']]]
        values = [['English', 'French']]

        # WHEN
        input_ids_tensor, attention_mask_tensor, segment_ids_tensor, input_lengths = encode_input(question_spans, column_names, table_names, values, tokenizer, 512, 'cpu')

        question_span_lengths, column_token_lengths, table_token_lengths, values_lengths = input_lengths

        # THEN
        self.assertEqual(len(input_ids_tensor[0]), len(attention_mask_tensor[0]))
        self.assertEqual(len(input_ids_tensor[0]), len(segment_ids_tensor[0]))

        self.assertEqual(len(input_ids_tensor[0]), sum(question_span_lengths[0]) + sum(column_token_lengths[0]) + sum(table_token_lengths[0]) + sum(values_lengths[0]))

    def test_encode_input__empty_values(self):
        # GIVEN
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        question_spans = [[['find'], ['all'], ['table', 'player'], ["'"], ['column', 'name', 'first'], ['and'], ['column', 'name', 'last'], ['who'], ['have'], ['empty'], ['death'], ['record'], ['.']]]
        column_names = [[['count', 'number', 'many', '[SEP]'], ['player', 'id', '[SEP]'], ['year', '[SEP]'], ['game', 'num', '[SEP]'], ['game', 'id', '[SEP]'], ['team', 'id', '[SEP]'], ['league', 'id', '[SEP]'], ['gp', '[SEP]'], ['starting', 'po', '[SEP]'], ['g', 'all', '[SEP]'], ['g', '[SEP]'], ['g', 'batting', '[SEP]'], ['g', 'defense', '[SEP]'], ['g', 'p', '[SEP]'], ['g', 'c', '[SEP]'], ['g', '1b', '[SEP]'], ['g', '2b', '[SEP]'], ['g', '3b', '[SEP]'], ['g', 's', '[SEP]'], ['g', 'lf', '[SEP]'], ['g', 'cf', '[SEP]'], ['g', 'rf', '[SEP]'], ['g', 'of', '[SEP]'], ['g', 'dh', '[SEP]'], ['g', 'ph', '[SEP]'], ['g', 'pr', '[SEP]'], ['award', 'id', '[SEP]'], ['tie', '[SEP]'], ['note', '[SEP]'], ['point', 'won', '[SEP]'], ['point', 'max', '[SEP]'], ['vote', 'first', '[SEP]'], ['stint', '[SEP]'], ['g', '[SEP]'], ['ab', '[SEP]'], ['r', '[SEP]'], ['h', '[SEP]'], ['double', '[SEP]'], ['triple', '[SEP]'], ['hr', '[SEP]'], ['rbi', '[SEP]'], ['sb', '[SEP]'], ['c', '[SEP]'], ['bb', '[SEP]'], ['so', '[SEP]'], ['ibb', '[SEP]'], ['hbp', '[SEP]'], ['sh', '[SEP]'], ['sf', '[SEP]'], ['g', 'idp', '[SEP]'], ['round', '[SEP]'], ['college', 'id', '[SEP]'], ['po', '[SEP]'], ['inn', 'out', '[SEP]'], ['po', '[SEP]'], ['a', '[SEP]'], ['e', '[SEP]'], ['dp', '[SEP]'], ['pb', '[SEP]'], ['wp', '[SEP]'], ['zr', '[SEP]'], ['glf', '[SEP]'], ['gcf', '[SEP]'], ['grf', '[SEP]'], ['tp', '[SEP]'], ['yearid', '[SEP]'], ['votedby', '[SEP]'], ['ballot', '[SEP]'], ['needed', '[SEP]'], ['vote', '[SEP]'], ['inducted', '[SEP]'], ['category', '[SEP]'], ['needed', 'note', '[SEP]'], ['park', 'id', '[SEP]'], ['span', 'first', '[SEP]'], ['span', 'last', '[SEP]'], ['game', '[SEP]'], ['opening', '[SEP]'], ['attendance', '[SEP]'], ['inseason', '[SEP]'], ['w', '[SEP]'], ['l', '[SEP]'], ['rank', '[SEP]'], ['plyr', 'mgr', '[SEP]'], ['half', '[SEP]'], ['birth', 'year', '[SEP]'], ['birth', 'month', '[SEP]'], ['birth', 'day', '[SEP]'], ['birth', 'country', '[SEP]'], ['birth', 'state', '[SEP]'], ['birth', 'city', '[SEP]'], ['death', 'year', '[SEP]'], ['death', 'month', '[SEP]'], ['death', 'day', '[SEP]'], ['death', 'country', '[SEP]'], ['death', 'state', '[SEP]'], ['death', 'city', '[SEP]'], ['name', 'first', '[SEP]'], ['name', 'last', '[SEP]'], ['name', 'given', '[SEP]'], ['weight', '[SEP]'], ['height', '[SEP]'], ['bat', '[SEP]'], ['throw', '[SEP]'], ['debut', '[SEP]'], ['final', 'game', '[SEP]'], ['retro', 'id', '[SEP]'], ['bbref', 'id', '[SEP]'], ['park', 'name', '[SEP]'], ['park', 'alias', '[SEP]'], ['city', '[SEP]'], ['state', '[SEP]'], ['country', '[SEP]'], ['cg', '[SEP]'], ['sho', '[SEP]'], ['sv', '[SEP]'], ['ipouts', '[SEP]'], ['er', '[SEP]'], ['baopp', '[SEP]'], ['era', '[SEP]'], ['bk', '[SEP]'], ['bfp', '[SEP]'], ['gf', '[SEP]'], ['salary', '[SEP]'], ['name', 'full', '[SEP]'], ['team', 'id', 'winner', '[SEP]'], ['league', 'id', 'winner', '[SEP]'], ['team', 'id', 'loser', '[SEP]'], ['league', 'id', 'loser', '[SEP]'], ['win', '[SEP]'], ['loss', '[SEP]'], ['tie', '[SEP]'], ['franchise', 'id', '[SEP]'], ['div', 'id', '[SEP]'], ['ghome', '[SEP]'], ['div', 'win', '[SEP]'], ['wc', 'win', '[SEP]'], ['lg', 'win', '[SEP]'], ['w', 'win', '[SEP]'], ['ra', '[SEP]'], ['ha', '[SEP]'], ['hra', '[SEP]'], ['bba', '[SEP]'], ['soa', '[SEP]'], ['fp', '[SEP]'], ['name', '[SEP]'], ['park', '[SEP]'], ['bpf', '[SEP]'], ['ppf', '[SEP]'], ['team', 'id', 'br', '[SEP]'], ['team', 'id', 'lahman45', '[SEP]'], ['team', 'id', 'retro', '[SEP]'], ['franchise', 'name', '[SEP]'], ['active', '[SEP]'], ['na', 'assoc', '[SEP]']]]
        table_names = [[['all', 'star'], ['appearance'], ['manager', 'award'], ['player', 'award'], ['manager', 'award', 'vote'], ['player', 'award', 'vote'], ['batting'], ['batting', 'postseason'], ['player', 'college'], ['fielding'], ['fielding', 'outfield'], ['fielding', 'postseason'], ['hall', 'of', 'fame'], ['home', 'game'], ['manager'], ['manager', 'half'], ['player'], ['park'], ['pitching'], ['pitching', 'postseason'], ['salary'], ['college'], ['postseason'], ['team'], ['team', 'franchise'], ['team', 'half']]]
        values = [['']]

        # WHEN
        input_ids_tensor, attention_mask_tensor, segment_ids_tensor, input_lengths = encode_input(question_spans, column_names, table_names, values, tokenizer, 512, 'cpu')

        question_span_lengths, column_token_lengths, table_token_lengths, values_lengths = input_lengths

        # THEN
        self.assertEqual(2, values_lengths[0][0])

    def test__tokenize_values(self):
        # GIVEN
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        values = ['English', 'French']

        # WHEN
        all_value_tokens, value_token_lengths, segment_ids = _tokenize_values(values, tokenizer)

        # THEN
        self.assertEqual(['english', '[SEP]', 'french', '[SEP]'], all_value_tokens)
        self.assertEqual([2, 2], value_token_lengths)
        self.assertEqual([1, 0, 1, 0], segment_ids)

    def test__tokenize_values__subword_tokenization(self):
        # GIVEN
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        values = ['Protoporphyrinogen IX', 'dummy']

        # WHEN
        all_value_tokens, value_token_lengths, segment_ids = _tokenize_values(values, tokenizer)

        # THEN
        self.assertEqual(['proto', '##por', '##phy', '##rino', '##gen', 'ix', '[SEP]', 'dummy', '[SEP]'], all_value_tokens)
        self.assertEqual([7, 2], value_token_lengths)
        self.assertEqual([1, 1, 1, 1, 1, 1, 0, 1, 0], segment_ids)

    def test__tokenize_values__integer(self):
        # GIVEN
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        values = [4]

        # WHEN
        all_value_tokens, value_token_lengths, segment_ids = _tokenize_values(values, tokenizer)

        # THEN
        self.assertEqual(['4', '[SEP]'], all_value_tokens)
        self.assertEqual([1, 0], segment_ids)

        ids = tokenizer.convert_tokens_to_ids(all_value_tokens)

        # make sure the token is not an unknown token
        self.assertNotEqual(tokenizer.unk_token_id, ids[0])

    def test__tokenize_values__float(self):
        # GIVEN
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        values = ['4.5']

        # WHEN
        all_value_tokens, value_token_lengths, segment_ids = _tokenize_values(values, tokenizer)

        # THEN
        self.assertEqual(['4', '.', '5', '[SEP]'], all_value_tokens)

        ids = tokenizer.convert_tokens_to_ids(all_value_tokens)

        # make sure the token is not an unknown token
        self.assertNotEqual(tokenizer.unk_token_id, ids[0])

    def test__tokenize_values__date(self):
        # GIVEN
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        values = ['2002-06-21']

        # WHEN
        all_value_tokens, value_token_lengths, segment_ids = _tokenize_values(values, tokenizer)

        # THEN
        self.assertEqual(['2002', '-', '06', '-', '21', '[SEP]'], all_value_tokens)

        ids = tokenizer.convert_tokens_to_ids(all_value_tokens)

        # make sure no token is unknown
        self.assertListEqual([], list(filter(lambda e: e == tokenizer.unk_token_id, ids)))

    def test__tokenize_values__fuzzy_strings(self):
        # GIVEN
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        values = ['%superstar%']

        # WHEN
        all_value_tokens, value_token_lengths, segment_ids = _tokenize_values(values, tokenizer)

        # THEN
        self.assertEqual(['%', 'superstar', '%', '[SEP]'], all_value_tokens)

        ids = tokenizer.convert_tokens_to_ids(all_value_tokens)

        # make sure no token is unknown
        self.assertListEqual([], list(filter(lambda e: e == tokenizer.unk_token_id, ids)))

    def test__tokenize_values__empty_string(self):
        # GIVEN
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        values = ['']

        # WHEN
        all_value_tokens, value_token_lengths, segment_ids = _tokenize_values(values, tokenizer)

        # THEN
        self.assertEqual(['empty', '[SEP]'], all_value_tokens)

        ids = tokenizer.convert_tokens_to_ids(all_value_tokens)

        # make sure no token is unknown
        self.assertListEqual([], list(filter(lambda e: e == tokenizer.unk_token_id, ids)))