import torch
from transformers import BertConfig, BertModel, BertTokenizer

from model.encoder.temporary.bert_enc import Bert_Layer
from model.encoder.encoder import TransformerEncoder
from utils import setup_device, set_seed_everywhere

############### NOTES TO REPRODUCE SAME OUTPUT AS Achronferry-implementation ###########################
# 1. make sure there are no words in the inputs which gets splitted into subwords (e.g. "goalie") ---> achronferry is not using subwords.
# 2. make sure you initialize the layers (especially the linear layer and the LSTM's) with the same seed. To do this, use "torch.manual_seed(0)" in the __init__ before each constructor.
# 3. make sure you use the "simple" transformer model call ("last_hidden_states, pooling_output = self.transformer_model(input_ids_tensor)") and not the advanced one with e.g. segment_id


config_class, model_class, tokenizer_class = (BertConfig, BertModel, BertTokenizer)
config = config_class.from_pretrained("bert-base-uncased")
tokenizer = tokenizer_class.from_pretrained("bert-base-uncased")

question = [['what'], ['are'], ['name'], ['of'], ['all'], ['column', 'state'], ['with'], ['table', 'college'], ['student'], ['playing'], ['in'], ['mid'], ['position'], ['but'], ['no'], ['goal'], ['?']]
question2 = [['what'], ['are'], ['name'], ['of'], ['all'], ['column', 'state'], ['with'], ['table', 'college'], ['student'], ['playing'], ['in'], ['mid'], ['position'], ['?']]
tables = [['college'], ['player'], ['try']]
tables2 = [['college'], ['player']]
columns = [['count', 'number', 'many'], ['college', 'name'], ['state'], ['enrollment'], ['player', 'id'], ['player', 'name'], ['yes', 'card'], ['training', 'hour']]
columns2 = [['count', 'number', 'many'], ['college', 'name'], ['state'], ['enrollment'], ['player', 'id'], ['player', 'name'], ['yes', 'card'], ['training', 'hour'], ['player', 'position'], ['decision']]

device, n_gpu = setup_device()
set_seed_everywhere(42, n_gpu)

encoder = TransformerEncoder('bert-base-uncased', device, 512, 300, 300)
encoder.to(device)

last_layer = encoder([question, question2], [columns, columns2], [tables, tables2])

args = type('', (object,), {"cuda": True,
                            "hidden_size": 300,
                            "col_embed_size": 300
                            })()

encoder_2 = Bert_Layer(args)
encoder_2.to(device)

last_layer_2 = encoder_2([question, question2], [len(question), len(question2)], [columns, columns2], [tables, tables2])

for output1, output2 in zip(last_layer, last_layer_2):
    print(torch.eq(output1, output2).all())