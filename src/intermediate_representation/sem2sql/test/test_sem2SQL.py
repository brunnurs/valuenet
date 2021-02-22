import json
from unittest import TestCase

from intermediate_representation.sem2sql.sem2SQL import transform
from intermediate_representation.sem_utils import alter_column0
from manual_inference.helper import get_schemas_spider, get_schemas_cordis


class Test(TestCase):
    def setUp(self):
        _, self.schemas, _, _ = get_schemas_spider()

    def test_transform__AND_OR_filter(self):
        # GIVEN
        with open('src/intermediate_representation/sem2sql/test/sql2SemQL_data/AND_OR.json', 'r', encoding='utf8') as f:
            example = json.load(f)

        example['model_result'] = example['rule_label']

        # WHEN
        alter_column0([example])
        result = transform(example, self.schemas[example['db_id']])

        # THEN
        self.assertEqual('''
SELECT count( *) 
FROM Student AS T1 
JOIN Has_Allergy AS T2 ON T1.StuID = T2.StuID 
WHERE T1.Sex = 'F' and T2.Allergy = 'Milk' 
or T2.Allergy = 'Eggs'
        '''.replace('\n', '').strip(), result[0].strip())

    def test_transform__complex_filter_structure(self):
        # GIVEN
        with open('src/intermediate_representation/sem2sql/test/sql2SemQL_data/four_AND_one_OR.json', 'r', encoding='utf8') as f:
            example = json.load(f)

        example['model_result'] = example['rule_label']

        # WHEN
        alter_column0([example])
        result = transform(example, self.schemas[example['db_id']])

        # THEN
        self.assertEqual('''
SELECT T1.name 
FROM business AS T1 
JOIN category AS T2 ON T1.business_id = T2.business_id 
JOIN neighbourhood AS T3 ON T1.business_id = T3.business_id 
WHERE T1.city = 'Madison' and T2.category_name = 'Italian' and T2.category_name = 'restaurant' and T3.neighbourhood_name = 'Meadowood' 
or T1.city = 'Soho' and T2.category_name = 'French' and T2.category_name = 'restaurant' and T3.neighbourhood_name = 'New York' 
        '''.replace('\n', '').strip(), result[0].strip())

    def test_transform__joins_modeled_as_nested(self):
        """
        SemQL has the issue that joins are not modeled explicitly - which makes sense as a user most probably will not mention them. Still though there
        are situations where there are important - for example when a join will restrict the resulting dataset as only samples with relations in both tables
        will be showed.
        To account for that we model such joins in SemQL as nested queries: SELECT XY FROM table_A WHERE id in (SELECT id FROM table_B).

        Even though this is not always 100% correct (what we do is a SEMI JOIN which has a slightly different behaviour) it solves the problem quite well.

        x§This example has is an extreme version of this rule.
        """
        # GIVEN
        with open('src/intermediate_representation/sem2sql/test/sql2SemQL_data/joins_modeled_as_nested_queries.json', 'r', encoding='utf8') as f:
            example = json.load(f)

        example['model_result'] = example['rule_label']

        # WHEN
        alter_column0([example])
        result = transform(example, self.schemas[example['db_id']])

        # THEN
        self.assertEqual('''
SELECT T1.name 
FROM tracks AS T1 
JOIN invoice_lines AS T4 ON T1.id = T4.track_id 
JOIN invoices AS T3 ON T4.invoice_id = T3.id 
JOIN customers AS T2 ON T3.customer_id = T2.id 
WHERE T2.first_name = 'Daan' 
and T2.last_name = 'Peeters' 
        '''.replace('\n', '').strip(), result[0].strip())

    def test_transform__temp(self):
        # GIVEN
        _, self.schemas, _, _ = get_schemas_cordis()

        with open('src/intermediate_representation/sem2sql/test/sql2SemQL_data/temp.json', 'r',
                  encoding='utf8') as f:
            example = json.load(f)

        example['model_result'] = example['rule_label']

        # WHEN
        alter_column0([example])
        result = transform(example, self.schemas[example['db_id']])

        # THEN
        print(result[0].strip())
