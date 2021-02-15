import json
import re
from unittest import TestCase

from preprocessing.sql2SemQL import Parser


def _clean(text):
    text = text.strip()
    return re.sub(r'\s+', ' ', text)


class TestParser(TestCase):
    def test_full_parse__AND_OR(self):
        # GIVEN
        with open('src/preprocessing/test/sql2SemQL_data/AND_OR.json', 'r', encoding='utf8') as f:
            example = json.load(f)

        # WHEN
        parser = Parser(example['values'])
        semql_result = parser.full_parse(example)

        # THEN
        actual = ' '.join([str(x) for x in semql_result])

        # One could argue also here if the most outer condition should not be an "OR" instead of an "AND".
        # See test "test_full_parse__four_AND_one_OR()" for an explanation.
        self.assertEqual(_clean('''
        Root1(3) 
            Root(3) 
                Sel(0) 
                    N(0) 
                        A(3) C(0) T(1) 
                Filter(0) 
                    Filter(2) 
                        A(0) C(8) T(2) V(0) 
                    Filter(1) 
                        Filter(2) 
                            A(0) C(4) T(1) V(1) 
                        Filter(2) 
                            A(0) C(4) T(1) V(2)
        '''), actual)

    def test_full_parse__two_AND(self):
        # GIVEN
        with open('src/preprocessing/test/sql2SemQL_data/two_AND.json', 'r', encoding='utf8') as f:
            example = json.load(f)

        # WHEN
        parser = Parser(example['values'])
        semql_result = parser.full_parse(example)

        # THEN
        actual = ' '.join([str(x) for x in semql_result])

        self.assertEqual(_clean('''
        Root1(3) 
            Root(3) 
                Sel(0) 
                    N(0) 
                    A(0) C(22) T(3) 
                Filter(0) 
                    Filter(2) 
                        A(0) C(21) T(3) V(2) 
                    Filter(0) 
                        Filter(3) 
                            A(0) C(43) T(3) V(3) 
                        Filter(3) A(0) C(43) T(3) V(1)
        '''), actual)

    def test_full_parse__three_AND(self):
        # GIVEN
        with open('src/preprocessing/test/sql2SemQL_data/three_AND.json', 'r', encoding='utf8') as f:
            example = json.load(f)

        # WHEN
        parser = Parser(example['values'])
        semql_result = parser.full_parse(example)

        # THEN
        actual = ' '.join([str(x) for x in semql_result])

        self.assertEqual(_clean('''
        Root1(3) 
            Root(3) 
                Sel(0) 
                    N(0) 
                        A(0) C(3) T(0) 
                Filter(0) 
                    Filter(2) 
                        A(0) C(5) T(0) V(2) 
                        Filter(0) 
                            Filter(2) 
                                A(0) C(13) T(1) V(0)
                                Filter(0)
                                    Filter(2) 
                                        A(0) C(13) T(1) V(3)
                                    Filter(2) 
                                        A(0) C(19) T(4) V(1)
        '''), actual)

    def test_full_parse__four_AND_one_OR(self):
        # GIVEN
        with open('src/preprocessing/test/sql2SemQL_data/four_AND_one_OR.json', 'r', encoding='utf8') as f:
            example = json.load(f)

        # WHEN
        parser = Parser(example['values'])
        semql_result = parser.full_parse(example)

        # THEN
        actual = ' '.join([str(x) for x in semql_result])

        # TODO: you would mabye expect here rather something like this: the filter clearly separated by an OR, then in
        # TODO: both sub-filters all the AND concatenation. Unfortunately the SemQL2SQL module is not yet able to handle
        # TODO: such a construct. It would though get produced exactly like this if you change the AND/OR order in _parse_filter_internal()
        # TODO: Therefore: Rewrite the SemQL2SQL module to handle this here properly.
        # self.assertEqual(_clean('''
        # Root1(3)
        #     Root(3)
        #         Sel(0)
        #             N(0)
        #                 A(0) C(3) T(0)
        #         Filter(1)
        #             Filter(0)
        #                 Filter(2)
        #                     A(0) C(5) T(0) V(2)
        #                     Filter(0)
        #                         Filter(2)
        #                             A(0) C(13) T(1) V(0)
        #                             Filter(0)
        #                                 Filter(2)
        #                                     A(0) C(13) T(1) V(3)
        #                                 Filter(2)
        #                                     A(0) C(19) T(4) V(1)
        #             Filter(0)
        #                 Filter(2)
        #                     A(0) C(5) T(0) V(4)
        #                     Filter(0)
        #                         Filter(2)
        #                             A(0) C(13) T(1) V(6)
        #                             Filter(0)
        #                                 Filter(2)
        #                                     A(0) C(13) T(1) V(3)
        #                                 Filter(2)
        #                                     A(0) C(19) T(4) V(5)
        # '''), actual)

        # What we get now is more simple: it is the one-to-one representation of the "WHERE" list in the "sql" variable.
        # Event though it looks wrong regarding the precedence, it will work when translating to SQL as the SemQL2sql module
        # is building up the SQL representation in the exact same order as we see it here.
        self.assertEqual(_clean('''
        Root1(3)
            Root(3)
                Sel(0)
                    N(0)
                        A(0) C(3) T(0)
                Filter(0)
                    Filter(2)
                        A(0) C(5) T(0) V(2)
                    Filter(0)
                        Filter(2)
                            A(0) C(13) T(1) V(0)
                        Filter(0)
                            Filter(2)
                                A(0) C(13) T(1) V(3)
                            Filter(0)
                                Filter(1)
                                    Filter(2)
                                        A(0) C(19) T(4) V(1)
                                    Filter(2)
                                        A(0) C(5) T(0) V(4)
                                Filter(0)
                                    Filter(2)
                                        A(0) C(13) T(1) V(6)
                                    Filter(0)
                                        Filter(2)
                                            A(0) C(13) T(1) V(3)
                                        Filter(2)
                                            A(0) C(19) T(4) V(5)
        '''), actual)


    def test_full_parse__build_up_values_on_the_fly(self):
        # GIVEN
        with open('src/preprocessing/test/sql2SemQL_data/three_AND.json', 'r', encoding='utf8') as f:
            example = json.load(f)

        parser = Parser(build_value_list=True)

        # WHEN
        # We don't care about the values here - we are just interested in the values-list
        _ = parser.full_parse(example)

        # THEN
        self.assertEqual(['Madison', 'Italian', 'restaurant', 'Meadowood'], parser.values)