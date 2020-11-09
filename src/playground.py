import json

# with open('data/spider/preprocessed_with_values.json', 'r', encoding='utf-8') as json_file:
#     data = json.load(json_file)
#     for row in data:
#         values = row['values']
#         if values:
#             candidates = row['ner_extracted_values_processed']
#             print(f'Values: {values}          Candiates: {candidates}')

values = [('F', 'src_ap', 'routes'), ('F', 'dst_ap', 'routes'), ('John F Kennedy International Airport', 'name', 'airports'), ('F', 'airline', 'routes'), ('INTERNACIONAL', 'callsign', 'airlines'), ('Denver International Airport', 'name', 'airports'), ('Kent International Airport', 'name', 'airports'), ('John F Kennedy International Airport', 'name', 'airports'), ('Yap International Airport', 'name', 'airports'), ('MBS International Airport', 'name', 'airports'), ('Gan International Airport', 'name', 'airports'), ('Ufa International Airport', 'name', 'airports'), ('Key West International Airport', 'name', 'airports'), ('Rivne International Airport', 'name', 'airports'), ('Chennai International Airport', 'name', 'airports'), ('Senai International Airport', 'name', 'airports'), ('Juneau International Airport', 'name', 'airports'), ('Benina International Airport', 'name', 'airports'), ('Gander International Airport', 'name', 'airports'), ('Vienna International Airport', 'name', 'airports'), ('Cuneo International Airport', 'name', 'airports'), ('Cassidy International Airport', 'name', 'airports'), ('Conakry International Airport', 'name', 'airports'), ('Kansai International Airport', 'name', 'airports'), ('Laredo International Airport', 'name', 'airports'), ('Nadi International Airport', 'name', 'airports'), ('Jinnah International Airport', 'name', 'airports'), ('Kaunas International Airport', 'name', 'airports'), ('Brunei International Airport', 'name', 'airports'), ('Juanda International Airport', 'name', 'airports'), ('Entebbe International Airport', 'name', 'airports'), ('Kelowna International Airport', 'name', 'airports'), ('Penang International Airport', 'name', 'airports'), ('Valley International Airport', 'name', 'airports'), ('Minsk National Airport', 'name', 'airports'), ('Kempegowda International Airport', 'name', 'airports'), ('International AirLink', 'name', 'airlines'), ('Mati National Airport', 'name', 'airports'), ('Lviv International Airport', 'name', 'airports'), ('Arad International Airport', 'name', 'airports'), ('Rota International Airport', 'name', 'airports'), ('Beja International Airport', 'name', 'airports'), ('Taba International Airport', 'name', 'airports'), ('Ovda International Airport', 'name', 'airports'), ('Malé International Airport', 'name', 'airports'), ('Jeju International Airport', 'name', 'airports'), ('Aden International Airport', 'name', 'airports'), ('Pisa International Airport', 'name', 'airports'), ('Niue International Airport', 'name', 'airports'), ('Muan International Airport', 'name', 'airports')]
for v, _, _ in values:
    print(f'"{v}",')

# tokenized = tokenizer.encode_plus(text=question,
#                               text_pair=columns + tables,
#                               add_special_tokens=True,
#                               max_length=100,
#                               truncation_strategy='do_not_truncate',
#                               pad_to_max_length=True)

# encoded = encode_input([question, question2], [columns, columns], [tables, tables], tokenizer, 551, 'cpu')
#
# # print(encoded)
# device, n_gpu = setup_device()
# set_seed_everywhere(42, n_gpu)
#
# encoder = TransformerEncoder('bert-base-uncased', device, 512, 300, 300)
# encoder.to(device)
#
# last_layer = encoder([question, question2], [columns, columns2], [tables, tables2])
#
# print(last_layer)

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


from more_itertools import flatten
from transformers import BartConfig, BartModel, BartTokenizer

config_class, model_class, tokenizer_class = (BartConfig, BartModel, BartTokenizer)

transformer_config = config_class.from_pretrained('facebook/bart-base')
tokenizer = tokenizer_class.from_pretrained('facebook/bart-base')
transformer_model = model_class.from_pretrained('facebook/bart-base', config=transformer_config)

tokens = ['Which', 'player', 'has', 'the', 'most', 'all', 'star', 'game', 'exper', 'iences', '?', 'Give', 'me', 'the', 'first', 'name', ',', 'last', 'name', 'and', 'id', 'of', 'the', 'player', ',', 'as', 'well', 'as', 'the', 'number', 'of', 'times', 'the', 'player', 'part', 'icipated', 'in', 'all', 'star', 'game', '.']
full_question = 'Which player has the most all star game experiences ? Give me the first name , last name and id of the player , as well as the number of times the player participated in all star game .'
question_untokenized = [['Which'], ['player'], ['has'], ['the'], ['most'], ['all'], ['star'], ['game'], ['experiences'], ['?'], ['Give'], ['me'], ['the'], ['first'], ['name'], [','], ['last'], ['name'], ['and'], ['id'], ['of'], ['the'], ['player'], [','], ['as'], ['well'], ['as'], ['the'], ['number'], ['of'], ['times'], ['the'], ['player'], ['participated'], ['in'], ['all'], ['star'], ['game'], ['.']]


tokenized = tokenizer(tokens, is_split_into_words=True)
tokenized2 = tokenizer(list(flatten(question_untokenized)), text_pair=list(flatten(question_untokenized)), is_split_into_words=True)
tokenized3 = tokenizer(full_question, text_pair=full_question)

print('ddd')