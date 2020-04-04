import torch
from transformers import BertConfig, BertModel, BertTokenizer

from model.encoder.encoder import TransformerEncoder
from utils import setup_device, set_seed_everywhere

config_class, model_class, tokenizer_class = (BertConfig, BertModel, BertTokenizer)
config = config_class.from_pretrained("bert-base-uncased")
tokenizer = tokenizer_class.from_pretrained("bert-base-uncased")

question = [['what'], ['are'], ['name'], ['of'], ['all'], ['column', 'state'], ['with'], ['table', 'college'], ['student'], ['playing'], ['in'], ['mid'], ['position'], ['but'], ['no'], ['goalie'], ['?']]
question2 = [['what'], ['are'], ['name'], ['of'], ['all'], ['column', 'state'], ['with'], ['table', 'college'], ['student'], ['playing'], ['in'], ['mid'], ['position'], ['?']]
tables = [['college'], ['player'], ['tryout']]
tables2 = [['college'], ['player']]
columns = [['count', 'number', 'many'], ['college', 'name'], ['state'], ['enrollment'], ['player', 'id'], ['player', 'name'], ['yes', 'card'], ['training', 'hour']]
columns2 = [['count', 'number', 'many'], ['college', 'name'], ['state'], ['enrollment'], ['player', 'id'], ['player', 'name'], ['yes', 'card'], ['training', 'hour'], ['player', 'position'], ['decision']]

# tokenized = tokenizer.encode_plus(text=question,
#                               text_pair=columns + tables,
#                               add_special_tokens=True,
#                               max_length=100,
#                               truncation_strategy='do_not_truncate',
#                               pad_to_max_length=True)

# encoded = encode_input([question, question2], [columns, columns], [tables, tables], tokenizer, 551, 'cpu')
#
# # print(encoded)
device, n_gpu = setup_device()
set_seed_everywhere(42, n_gpu)

encoder = TransformerEncoder('bert-base-uncased', device, 512, 300, 300)
encoder.to(device)

last_layer = encoder([question, question2], [columns, columns2], [tables, tables2])

print(last_layer)

# import json
#
# table_names_baseball = [
#     "all_star",
#     "appearances",
#     "manager_award",
#     "player_award",
#     "manager_award_vote",
#     "player_award_vote",
#     "batting",
#     "batting_postseason",
#     "player_college",
#     "fielding",
#     "fielding_outfield",
#     "fielding_postseason",
#     "hall_of_fame",
#     "home_game",
#     "manager",
#     "manager_half",
#     "player",
#     "park",
#     "pitching",
#     "pitching_postseason",
#     "salary",
#     "college",
#     "postseason",
#     "team",
#     "team_franchise",
#     "team_half"
# ]
#
# table_occurances = {key: 0 for key in table_names_baseball}
#
#
# with open('data/spider/train.json', 'r', encoding='utf8') as f:
#     train_samples = json.load(f)
#
#     for sample in train_samples:
#         if sample["db_id"] == "baseball_1":
#             print(sample["question"])
#             print(sample["query"])
#             print()
#
#             for token in sample["query_toks"]:
#                 if token in table_occurances.keys():
#                     table_occurances[token] = table_occurances[token] + 1
#
#     print(table_occurances)
