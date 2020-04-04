baseball_1_tables = [['all', 'star'], ['appearance'], ['manager', 'award'], ['player', 'award'],
                     ['manager', 'award', 'vote'],
                     ['player', 'award', 'vote'], ['batting'], ['batting', 'postseason'], ['player', 'college'],
                     ['fielding'],
                     ['fielding', 'outfield'], ['fielding', 'postseason'], ['hall', 'of', 'fame'], ['home', 'game'],
                     ['manager'],
                     ['manager', 'half'], ['player'], ['park'], ['pitching'], ['pitching', 'postseason'], ['salary'],
                     ['college'],
                     ['postseason'], ['team'], ['team', 'franchise'], ['team', 'half']]


def _remove_basesball_1(column_names, table_names):
    """
    We just strip away enough unused columns to get below the 512 token limit of most transformers.
    """
    if table_names == baseball_1_tables:
        # columns from table "pitching_postseason" which get never used in queries
        column_names.remove(['w'])
        column_names.remove(['l'])
        column_names.remove(['g'])
        column_names.remove(['cg'])
        column_names.remove(['sho'])
        column_names.remove(['sv'])
        column_names.remove(['ipouts'])
        column_names.remove(['h'])
        column_names.remove(['hr'])
        column_names.remove(['bb'])
        column_names.remove(['so'])
        column_names.remove(['baopp'])
        column_names.remove(['era'])
        column_names.remove(['ibb'])
        column_names.remove(['wp'])
        column_names.remove(['bk'])
        column_names.remove(['bfp'])
        column_names.remove(['gf'])
        column_names.remove(['r'])
        column_names.remove(['sh'])
        column_names.remove(['sf'])
        column_names.index(['g', 'idp'])

        # columns from table "fielding" which get never used in queries
        column_names.index(['po'])
        column_names.index(['a'])
        column_names.index(['e'])
        column_names.index(['dp'])
        column_names.index(['pb'])
        column_names.index(['sb'])
        column_names.index(['zr'])

    return column_names, table_names


def prune_large_database_schemas(column_names, table_names):
    """
    There are some schemas (right now only "baseball_1") which are huge and contain large amount of tables & columns never used in the question.
    Here we prune the those schemas.

    NOTE URSIN: This pre-processing step will become more important and should be part of the proper pre-processing. It will include:
    - re-naming technical table/columns. Create a mapping which fits better to the words a user would use (or even use word-embeddings representing a table/schema).
    - remove technical parts of the schema. Simplify the schema to a level a user would understand it. This is very specific to the database AND user group.
    """

    return _remove_basesball_1(column_names, table_names)
