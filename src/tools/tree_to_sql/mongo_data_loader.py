import json

from pymongo import MongoClient

from tools.tree_to_sql.tree_to_sql import tree_to_sql


def main():
    # ATTENTION: Make sure you open the ssh tunnel before running this!
    # ssh -vvv -L 127.0.0.1:8888:0.0.0.0:27017 -N brrn@biosoda.cloudlab.zhaw.ch
    client = MongoClient('mongodb://soda:123@localhost:8888/semql')

    db = client.semql

    print(db.list_collection_names())
    tree_collection = db['First_cordisv02_labelled_trees']

    samples = []
    for idx, tree in enumerate(tree_collection.find()):
        print(f"{idx} User:'{tree['user_name']}' Question:'{tree['question']}'")
        sql = tree_to_sql(tree['tree'])
        print(sql)
        print()
        samples.append({
            'db_id': "cordis_temporary",
            'id': str(tree['_id']),
            'user': tree['user_name'],
            'question': tree['question'],
            'query': sql
        })

    with open('data/cordis/trees/all.json', 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2)


if __name__ == '__main__':
    main()
