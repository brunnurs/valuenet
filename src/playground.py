from transformers import BertConfig, BertModel, BertTokenizer

from src.model.encoder.encoder import TransformerEncoder
from src.model.encoder.input_features import encode_input
from src.utils import setup_device, set_seed_everywhere

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
#
encoder = TransformerEncoder('bert-base-uncased', device, 512, 300, 300)
encoder.to(device)

last_layer = encoder([question, question2], [columns, columns2], [tables, tables2])
print(last_layer)