import argparse
import json
import os
import sys

from preprocessing.utils import load_dataSets, find_table_of_star_column
from intermediate_representation.semQL import Root1, Root, N, A, C, T, Sel, Sup, Filter, Order, V

sys.path.append("..")

##### NOTE: this code is just lazily copied and we only need it right now to filter out the value from the SQL-construct #####
#TODO reduce the code to get the values only!

class Parser:

    def __init__(self) -> None:
        # this list will contain all the values in the filter statements. It will be part of the pre-processed data afterwards.
        self.values = []

    def _parse_root(self, sql):
        """
        parsing the sql by the grammar
        R ::= Select | Select Filter | Select Order | ... |
        :return: [R(), states]
        """
        use_sup, use_ord, use_fil = True, True, False

        if sql['sql']['limit'] == None:
            use_sup = False

        if sql['sql']['orderBy'] == []:
            use_ord = False
        elif sql['sql']['limit'] != None:
            use_ord = False

        # check the where and having
        if sql['sql']['where'] != [] or \
                sql['sql']['having'] != []:
            use_fil = True

        if use_fil and use_sup:
            return [Root(0)], ['FILTER', 'SUP', 'SEL']
        elif use_fil and use_ord:
            return [Root(1)], ['ORDER', 'FILTER', 'SEL']
        elif use_sup:
            return [Root(2)], ['SUP', 'SEL']
        elif use_fil:
            return [Root(3)], ['FILTER', 'SEL']
        elif use_ord:
            return [Root(4)], ['ORDER', 'SEL']
        else:
            return [Root(5)], ['SEL']

    @staticmethod
    def _parser_column0(sql, select):
        table_idx = find_table_of_star_column(sql, select)
        return T(table_idx)

    def _parse_select(self, sql):
        """
        parsing the sql by the grammar
        Select ::= A | AA | AAA | ... |
        A ::= agg column table
        :return: [Sel(), states]
        """
        result = []
        is_distinct = sql['sql']['select'][0]  # is distinct on the whole select.
        select = sql['sql']['select'][1]

        # as a simplification we assume that if any of the columns is distinct, the whole query is distinct. This might be oversimplified, but
        # it is actually hard to find a way to phrase a real question where some columns are distinct and others not. And in the DEV set, there is also no such example, so we
        # simplified the SemQL language to that.
        if not is_distinct:
            is_distinct = any(sel[1][1][2] for sel in select)

        if is_distinct:
            result.append(Sel(1))
        else:
            result.append(Sel(0))

        result.append(N(len(select) - 1))  # N() encapsulates the number of columns. The -1 is used in case there is only one column to select: in that case, #0 of grammar_dict is used, which is 'N A'.

        for sel in select:
            result.append(A(sel[0])) # A() represents an aggregator. e.g. #0 is 'none', #3 is 'count'
            result.append(C(sql['col_set'].index(sql['names'][sel[1][1][1]])))
            # now check for the situation with *
            if sel[1][1][1] == 0:
                result.append(self._parser_column0(sql, select))  # The "*" needs an extra handling, as it belongs not to a "normal" table.
            else:
                result.append(T(sql['col_table'][sel[1][1][1]]))  # for every other column, we can simply add a T() with the table this column belongs to.

        return result, None

    def _parse_sup(self, sql):
        """
        parsing the sql by the grammar
        Sup ::= Most A | Least A
        A ::= agg column table
        :return: [Sup(), states]
        """
        result = []
        select = sql['sql']['select'][1]
        if sql['sql']['limit'] == None:
            return result, None
        if sql['sql']['orderBy'][0] == 'desc':
            result.append(Sup(0))
        else:
            result.append(Sup(1))

        result.append(A(sql['sql']['orderBy'][1][0][1][0]))
        result.append(C(sql['col_set'].index(sql['names'][sql['sql']['orderBy'][1][0][1][1]])))
        if sql['sql']['orderBy'][1][0][1][1] == 0:
            result.append(self._parser_column0(sql, select))
        else:
            result.append(T(sql['col_table'][sql['sql']['orderBy'][1][0][1][1]]))
        return result, None

    def _parse_filter(self, sql):
        """
        parsing the sql by the grammar
        Filter ::= and Filter Filter | ... |
        A ::= agg column table
        :return: [Filter(), states]
        """
        result = []
        # check the where
        if sql['sql']['where'] != [] and sql['sql']['having'] != []:
            result.append(Filter(0))

        if sql['sql']['where'] != []:
            # check the not and/or
            if len(sql['sql']['where']) == 1:
                result.extend(self.parse_one_condition(sql['sql']['where'][0], sql['names'], sql))
            elif len(sql['sql']['where']) == 3:
                if sql['sql']['where'][1] == 'or':
                    result.append(Filter(1))
                else:
                    result.append(Filter(0))
                result.extend(self.parse_one_condition(sql['sql']['where'][0], sql['names'], sql))
                result.extend(self.parse_one_condition(sql['sql']['where'][2], sql['names'], sql))
            else:
                if sql['sql']['where'][1] == 'and' and sql['sql']['where'][3] == 'and':
                    result.append(Filter(0))
                    result.extend(self.parse_one_condition(sql['sql']['where'][0], sql['names'], sql))
                    result.append(Filter(0))
                    result.extend(self.parse_one_condition(sql['sql']['where'][2], sql['names'], sql))
                    result.extend(self.parse_one_condition(sql['sql']['where'][4], sql['names'], sql))
                elif sql['sql']['where'][1] == 'and' and sql['sql']['where'][3] == 'or':
                    result.append(Filter(0))
                    result.extend(self.parse_one_condition(sql['sql']['where'][0], sql['names'], sql))
                    result.append(Filter(1))
                    result.extend(self.parse_one_condition(sql['sql']['where'][2], sql['names'], sql))
                    result.extend(self.parse_one_condition(sql['sql']['where'][4], sql['names'], sql))
                elif sql['sql']['where'][1] == 'or' and sql['sql']['where'][3] == 'and':
                    result.append(Filter(1))
                    result.extend(self.parse_one_condition(sql['sql']['where'][0], sql['names'], sql))
                    result.append(Filter(0))
                    result.extend(self.parse_one_condition(sql['sql']['where'][2], sql['names'], sql))
                    result.extend(self.parse_one_condition(sql['sql']['where'][4], sql['names'], sql))
                else:
                    result.append(Filter(1))
                    result.append(Filter(1))
                    result.extend(self.parse_one_condition(sql['sql']['where'][0], sql['names'], sql))
                    result.extend(self.parse_one_condition(sql['sql']['where'][2], sql['names'], sql))
                    result.extend(self.parse_one_condition(sql['sql']['where'][4], sql['names'], sql))

                # TODO: right now we don't handle queries which have more than 3 filters (so 3 filters and 2 AND/OR combinations).
                # The code above should be written in a more dynamic way


        # check having
        if sql['sql']['having'] != []:
            result.extend(self.parse_one_condition(sql['sql']['having'][0], sql['names'], sql))
        return result, None

    def _parse_order(self, sql):
        """
        parsing the sql by the grammar
        Order ::= asc A | desc A
        A ::= agg column table
        :return: [Order(), states]
        """
        result = []

        if 'order' not in sql['query_toks_no_value'] or 'by' not in sql['query_toks_no_value']:
            return result, None
        elif 'limit' in sql['query_toks_no_value']:
            return result, None
        else:
            if sql['sql']['orderBy'] == []:
                return result, None
            else:
                select = sql['sql']['select'][1]
                if sql['sql']['orderBy'][0] == 'desc':
                    result.append(Order(0))
                else:
                    result.append(Order(1))
                result.append(A(sql['sql']['orderBy'][1][0][1][0]))
                result.append(C(sql['col_set'].index(sql['names'][sql['sql']['orderBy'][1][0][1][1]])))
                if sql['sql']['orderBy'][1][0][1][1] == 0:
                    result.append(self._parser_column0(sql, select))
                else:
                    result.append(T(sql['col_table'][sql['sql']['orderBy'][1][0][1][1]]))
        return result, None

    def parse_one_condition(self, sql_condit, names, sql):
        result = []
        # check if V(root)
        nest_query = True
        if type(sql_condit[3]) != dict:
            nest_query = False

        if sql_condit[0] == True:
            if sql_condit[1] == 9:
                # not like only with values
                fil = Filter(10)
            elif sql_condit[1] == 8:
                # not in with Root
                fil = Filter(19)
            else:
                print(sql_condit[1])
                raise NotImplementedError("not implement for the others FIL")
        else:
            # check for Filter (<,=,>,!=,between, >=,  <=, ...)
            # Ursin: This map is a mapping between the index of the WHERE_OPS in spider and the Filter() index in SemQL:
            # WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
            # Filter --> see Filter-class
            # Example: 1:8 --> the filter type "between" is a 1 in the spider notation, but a 8 in SemQL.
            single_map = {1: 8, 2: 2, 3: 5, 4: 4, 5: 7, 6: 6, 7: 3}
            nested_map = {1: 15, 2: 11, 3: 13, 4: 12, 5: 16, 6: 17, 7: 14}
            if sql_condit[1] in [1, 2, 3, 4, 5, 6, 7]:
                if nest_query == False:
                    fil = Filter(single_map[sql_condit[1]])
                else:
                    fil = Filter(nested_map[sql_condit[1]])
            elif sql_condit[1] == 9:
                fil = Filter(9)
            elif sql_condit[1] == 8:
                fil = Filter(18)
            else:
                print(sql_condit[1])
                raise NotImplementedError("not implement for the others FIL")

        result.append(fil)

        result.append(A(sql_condit[2][1][0]))
        result.append(C(sql['col_set'].index(sql['names'][sql_condit[2][1][1]])))
        if sql_condit[2][1][1] == 0:
            select = sql['sql']['select'][1]
            result.append(self._parser_column0(sql, select))
        else:
            result.append(T(sql['col_table'][sql_condit[2][1][1]]))

        # This are filter statements which contain Values - we build up a value list and extend the SemQL AST with a "V" action.
        if 2 <= fil.id_c <= 10:
            val = sql_condit[3]
            if isinstance(val, str):
                val = val.strip('\'\"')   # remove string quotes, as we will add them later anyway.
            self.values.append(val)
            result.append((V(self.values.index(val))))

            # Filter(8) is the "X.Y BETWEEN A AND B" case - here we have to store an additional value.
            if fil.id_c == 8:
                val = sql_condit[4]
                if isinstance(val, str):
                    val = val.strip('\'\"')   # remove string quotes, as we will add them later anyway.
                self.values.append(val)
                result.append((V(self.values.index(val))))

        # check for the nested value
        if type(sql_condit[3]) == dict:
            nest_query = {}
            nest_query['names'] = names
            nest_query['query_toks_no_value'] = ""
            nest_query['sql'] = sql_condit[3]
            nest_query['col_table'] = sql['col_table']
            nest_query['col_set'] = sql['col_set']
            nest_query['table_names'] = sql['table_names']
            nest_query['question'] = sql['question']
            nest_query['query'] = sql['query']
            nest_query['keys'] = sql['keys']
            result.extend(self.parser(nest_query))

        return result

    def _parse_step(self, state, sql):

        if state == 'ROOT':
            return self._parse_root(sql)

        if state == 'SEL':
            return self._parse_select(sql)

        elif state == 'SUP':
            return self._parse_sup(sql)

        elif state == 'FILTER':
            return self._parse_filter(sql)

        elif state == 'ORDER':
            return self._parse_order(sql)
        else:
            raise NotImplementedError("Not the right state")

    def full_parse(self, query):
        """
        With this code we pare a SQL-query (as specified by spider) into a SemQL-AST
        """

        sql = query['sql']

        nest_query = {}
        nest_query['names'] = query['names']
        nest_query['query_toks_no_value'] = ""
        nest_query['col_table'] = query['col_table']
        nest_query['col_set'] = query['col_set']
        nest_query['table_names'] = query['table_names']
        nest_query['question'] = query['question']
        nest_query['query'] = query['query']
        nest_query['keys'] = query['keys']

        ### the following 3 if's are exception case, for "intersect", "union" and "except" cases. A normal query uses the [Root1(3)] and calls self.parser only once ###

        if sql['intersect']:
            results = [Root1(0)]
            nest_query['sql'] = sql['intersect']
            # an intersect query, similar to "union" and "except", will call the parser twice (once for each sub-query)
            results.extend(self.parser(query))
            results.extend(self.parser(nest_query))
            return results

        if sql['union']:
            results = [Root1(1)]
            nest_query['sql'] = sql['union']
            results.extend(self.parser(query))
            results.extend(self.parser(nest_query))
            return results

        if sql['except']:
            results = [Root1(2)]
            nest_query['sql'] = sql['except']
            results.extend(self.parser(query))
            results.extend(self.parser(nest_query))
            return results

        results = [Root1(3)]
        results.extend(self.parser(query))

        return results

    def parser(self, query):
        stack = ["ROOT"]
        result = []
        while len(stack) > 0:
            state = stack.pop()
            step_result, step_state = self._parse_step(state, query)
            result.extend(step_result)
            if step_state:
                stack.extend(step_state)
        return result


def format_groundtruth_value(val):
    if isinstance(val, str):
        val = val.strip('\'\"')  # remove string quotes, as we will add them later anyway.

        # if it is a fuzzy string (e.g. '%hello%') we wan't to remove the wildcards, as they get in later as part of the post-processing.
        if val.startswith('%') or val.endswith('%'):
            val = val.replace('%', '')

    # the ground truth values are all floats, even if there is no decimals (e.g. 56.0 instead of 56). But to make the
    # .index() work, we need exact matches!
    if isinstance(val, float) and val.is_integer():
        val = int(val)

    # results from NER will only be strings - therefore we need to make sure the values we use here are also string only!
    return str(val)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, help='dataset', required=True)
    arg_parser.add_argument('--table_path', type=str, help='table dataset', required=True)
    arg_parser.add_argument('--ner_path', type=str, help='file containing the values extracted by NER (e.g. ner_train.json)', required=True)
    args = arg_parser.parse_args()

    # loading dataSets
    data, table = load_dataSets(args)
    processed_data = []

    with open(os.path.join(args.ner_path), 'r', encoding='utf-8') as json_file:
        ner_data = json.load(json_file)

    if len(ner_data) != len(data):
        raise ValueError(f'There are {len(ner_data)} NER data rows and {len(data)} samples in the training set. Something is wrong!')

    for row, ner_row in zip(data, ner_data):

        if len(row['sql']['select'][1]) > 5:
            ner_row['values'] = ['Cant handle this sample as it has more than 5 select columns']
            continue

        parser = Parser()
        _ = parser.full_parse(row)

        values_formatted = [format_groundtruth_value(val) for val in parser.values]
        ner_row['values'] = values_formatted

        question = row['question']

        print(f'Found values {values_formatted} for question: "{question}"')

    print(f'Read out values from {len(data)} questions and added it to NER-file {args.ner_path}')
    with open(args.ner_path, 'w', encoding='utf8') as f:
        f.write(json.dumps(ner_data, indent=2))
