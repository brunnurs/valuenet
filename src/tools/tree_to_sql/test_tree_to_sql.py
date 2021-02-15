import json
from unittest import TestCase

from tools.tree_to_sql.tree_to_sql import tree_to_sql


class TestTreeToSql(TestCase):
    def test_ursin_query_2(self):
        # GIVEN
        with open('data/cordis/trees/5fa114c4913f25fdb2030f8f.json', 'r', encoding='utf8') as f:
            tree = json.load(f)

        # WHEN
        sql = tree_to_sql(tree)

        # THEN
        expected: str = '''
select avg(projects.total_cost)
from funding_schemes join projects on funding_schemes.code = projects.ec_fund_scheme
where funding_schemes.title != 'Cooperation network contracts'
and projects.start_date != '2012-06-25'
'''

        self.assertEqual(expected.strip(), sql.strip())

    def test_ursin_query_3(self):
        # GIVEN
        with open('data/cordis/trees/5fb2315b913f25fdb2030fc9.json', 'r', encoding='utf8') as f:
            tree = json.load(f)

        # WHEN
        sql = tree_to_sql(tree)

        # THEN
        expected: str = '''
select avg(projects.total_cost)
from eu_territorial_units join project_members on eu_territorial_units.geocode_regions = project_members.geocode_regions_3
 join projects on project_members.project = projects.unics_id
where eu_territorial_units.description = 'Sheffield'
and project_members.department_name != 'Agrotechnology and Food Science Group'
and projects.end_date <= '2010-11-17'
'''

        self.assertEqual(expected.strip(), sql.strip())

    def test_ursin_query_4(self):
        # GIVEN
        with open('data/cordis/trees/5fb234e7913f25fdb2030fca.json', 'r', encoding='utf8') as f:
            tree = json.load(f)

        # WHEN
        sql = tree_to_sql(tree)

        # THEN
        expected: str = '''
select count(distinct people.full_name)
from subject_areas join project_subject_areas on subject_areas.code = project_subject_areas.subject_area
 join projects on project_subject_areas.project = projects.unics_id
 join people on projects.principal_investigator = people.unics_id
where projects.acronym != 'CorticALS'
and subject_areas.title = 'Clean coal technologies'
'''

        self.assertEqual(expected.strip(), sql.strip())

    def test_ursin_query_5(self):
        # GIVEN
        with open('data/cordis/trees/5fb23762913f25fdb2030fcb.json', 'r', encoding='utf8') as f:
            tree = json.load(f)

        # WHEN
        sql = tree_to_sql(tree)

        # THEN
        expected: str = '''
select sum(projects.total_cost)
from subject_areas join project_subject_areas on subject_areas.code = project_subject_areas.subject_area
 join projects on project_subject_areas.project = projects.unics_id
where subject_areas.title = 'Food'
and projects.title = 'Innovative plasma based transformation of food waste into high value graphitic carbon and renewable hydrogen'
'''

        self.assertEqual(expected.strip(), sql.strip())

    def test_ursin_query_6(self):
        # GIVEN
        with open('data/cordis/trees/5fb23d0b913f25fdb2030fcc.json', 'r', encoding='utf8') as f:
            tree = json.load(f)

        # WHEN
        sql = tree_to_sql(tree)

        # THEN
        expected: str = '''
select not exists(select distinct people.full_name
from project_members join projects on project_members.project = projects.unics_id
 join activity_types on activity_types.code = project_members.activity_type
 join people on projects.principal_investigator = people.unics_id
where activity_types.description != 'Public bodies (excluding Research Organisations and Secondary or Higher Education Establishments)'
and projects.end_year != '2017')
'''

        self.assertEqual(expected.strip(), sql.strip())

    def test_ursin_query_7(self):
        # GIVEN
        with open('data/cordis/trees/5fb253d3913f25fdb2030fcd.json', 'r', encoding='utf8') as f:
            tree = json.load(f)

        # WHEN
        sql = tree_to_sql(tree)

        # THEN
        expected: str = '''
select count(distinct topics.title)
from project_topics join topics on project_topics.topic = topics.code
'''

        self.assertEqual(expected.strip(), sql.strip())

    def test_ursin_query_8(self):
        # GIVEN
        with open('data/cordis/trees/5fb255ba913f25fdb2030fce.json', 'r', encoding='utf8') as f:
            tree = json.load(f)

        # WHEN
        sql = tree_to_sql(tree)

        # THEN
        expected: str = '''
        select not exists(select distinct subject_areas.title
from projects join project_subject_areas on projects.unics_id = project_subject_areas.project
 join subject_areas on project_subject_areas.subject_area = subject_areas.code
where projects.start_year = '2018')
'''

        self.assertEqual(expected.strip(), sql.strip())

    def test_ursin_query_9(self):
        # GIVEN
        with open('data/cordis/trees/5fb2574a913f25fdb2030fcf.json', 'r', encoding='utf8') as f:
            tree = json.load(f)

        # WHEN
        sql = tree_to_sql(tree)

        # THEN
        expected: str = '''
select distinct countries.country_name
from institutions join countries on institutions.country_id = countries.unics_id
where institutions.institutions_name != 'CHECKLIGHT LTD'
'''

        self.assertEqual(expected.strip(), sql.strip())

    def test_ursin_query_10(self):
        # GIVEN
        with open('data/cordis/trees/5fb269c1913f25fdb2030fd0.json', 'r', encoding='utf8') as f:
            tree = json.load(f)

        # WHEN
        sql = tree_to_sql(tree)

        # THEN
        expected: str = '''
select avg(projects.total_cost)
from eu_territorial_units join project_members on eu_territorial_units.geocode_regions = project_members.geocode_regions_3
 join institutions on institutions.geocode_regions_3 = eu_territorial_units.geocode_regions
 join countries on countries.unics_id = institutions.country_id
 join projects on project_members.project = projects.unics_id
where countries.country_name = 'Wallis and Futuna'
and institutions.acronym = 'CED'
'''

        self.assertEqual(expected.strip(), sql.strip())

    def test_correct_order_of_joins(self):
        # GIVEN
        with open('data/cordis/trees/5ed78f2e059156a55033df76.json', 'r', encoding='utf8') as f:
            tree = json.load(f)

        # WHEN
        sql = tree_to_sql(tree)

        # THEN
        expected: str = '''
select count(distinct people.full_name)
from erc_panels join project_erc_panels on erc_panels.code = project_erc_panels.panel
 join projects on project_erc_panels.project = projects.unics_id
 join erc_research_domains on erc_research_domains.code = erc_panels.part_of
 join people on projects.principal_investigator = people.unics_id
where erc_research_domains.description = 'Social Sciences And Humanities'
and erc_panels.description = 'The Human Mind and its complexity'
and projects.total_cost < '2477105.6'
'''

        self.assertEqual(expected.strip(), sql.strip())
