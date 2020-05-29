import json
import matplotlib.pyplot as plt

with open('data/spider/original/tables.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

all_number_of_table_tokens = []
all_number_of_column_tokens = []
all_total_tokens = []

for schema in data:
    number_of_table_tokens = sum([len(t.split(' ')) for t in schema['table_names']])
    all_number_of_table_tokens.append(number_of_table_tokens)
    number_of_column_tokens = sum([len(c[1].split(' ')) for c in schema['column_names']])
    all_number_of_column_tokens.append(number_of_column_tokens)
    db = schema['db_id']
    all_total_tokens.append(number_of_table_tokens + number_of_column_tokens)

    print(f"db: {db}, table-tokens: {number_of_table_tokens}, column-tokens: {number_of_column_tokens}, total: {number_of_table_tokens + number_of_column_tokens}")

plt.hist(all_total_tokens, bins=range(10, 600, 10))
plt.show()

print(sorted(all_total_tokens))