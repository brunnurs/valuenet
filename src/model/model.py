import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
from torch.autograd import Variable

from intermediate_representation.beam import Beams, ActionInfo
from model.encoder.encoder import TransformerEncoder
from spider.example import Batch
import neural_network_utils as nn_utils
from model.basic_model import BasicModel
from model.pointer_net import PointerNet
from intermediate_representation import semQL as semQL


class IRNet(BasicModel):

    def __init__(self, args, device, grammar):
        super(IRNet, self).__init__()
        self.args = args
        self.grammar = grammar
        self.use_column_pointer = args.column_pointer

        if args.cuda:
            self.new_long_tensor = torch.cuda.LongTensor
            self.new_tensor = torch.cuda.FloatTensor
        else:
            self.new_long_tensor = torch.LongTensor
            self.new_tensor = torch.FloatTensor

        self.encoder = TransformerEncoder(args.encoder_pretrained_model,
                                          device, args.max_seq_length,
                                          args.embed_size,
                                          args.hidden_size)

        input_dim = args.action_embed_size + \
                    args.att_vec_size + \
                    args.type_embed_size
        # previous action
        # input feeding
        # pre type embedding

        self.lf_decoder_lstm = nn.LSTMCell(input_dim, args.hidden_size)

        self.sketch_decoder_lstm = nn.LSTMCell(input_dim, args.hidden_size)

        # initialize the decoder's state and cells with encoder hidden states
        self.decoder_cell_init = nn.Linear(self.encoder.encoder_hidden_size, args.hidden_size)

        self.att_sketch_linear = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.att_lf_linear = nn.Linear(args.hidden_size, args.hidden_size, bias=False)

        self.sketch_att_vec_linear = nn.Linear(args.hidden_size + args.hidden_size, args.att_vec_size, bias=False)
        self.lf_att_vec_linear = nn.Linear(args.hidden_size + args.hidden_size, args.att_vec_size, bias=False)

        self.prob_att = nn.Linear(args.att_vec_size, 1)
        self.prob_len = nn.Linear(1, 1)

        self.col_type = nn.Linear(4, args.col_embed_size)
        self.sketch_encoder = nn.LSTM(args.action_embed_size, args.action_embed_size // 2, bidirectional=True,
                                      batch_first=True)

        self.production_embed = nn.Embedding(len(grammar.prod2id), args.action_embed_size)
        self.type_embed = nn.Embedding(len(grammar.type2id), args.type_embed_size)
        self.production_readout_b = nn.Parameter(torch.FloatTensor(len(grammar.prod2id)).zero_())

        self.att_project = nn.Linear(args.hidden_size + args.type_embed_size, args.hidden_size)

        self.N_embed = nn.Embedding(len(semQL.N._init_grammar()), args.action_embed_size)

        self.read_out_act = torch.tanh if args.readout == 'non_linear' else nn_utils.identity

        self.query_vec_to_action_embed = nn.Linear(args.att_vec_size, args.action_embed_size,
                                                   bias=args.readout == 'non_linear')

        # production_readout is in the end a concatenation of 3 function:
        # first, we put the att_t (which is the input "q" here), through a linear layer ("query_vec_to_action_embed") with output-size "args.action_embed_size".
        # next, we put the result through a non-functional TanH or the identity function ("read_out_act"). By default we use identity.
        # then, we multiply the result of this again linearly with the production embeddings ("production_embed") and add a bias. That way we end up with a matrix of
        # dimensions (batch_size * len(production_embeddings)). We now have a value for each possible (next)-action, so we can do a softmax over it.

        self.production_readout = lambda q: F.linear(self.read_out_act(self.query_vec_to_action_embed(q)),
                                                     self.production_embed.weight, self.production_readout_b)

        self.column_rnn_input = nn.Linear(args.col_embed_size, args.action_embed_size, bias=False)
        self.table_rnn_input = nn.Linear(args.col_embed_size, args.action_embed_size, bias=False)
        self.value_rnn_input = nn.Linear(args.col_embed_size, args.action_embed_size, bias=False)


        self.dropout = nn.Dropout(args.dropout)

        self.column_pointer_net = PointerNet(args.hidden_size, args.col_embed_size, attention_type=args.column_att)

        self.table_pointer_net = PointerNet(args.hidden_size, args.col_embed_size, attention_type=args.column_att)

        self.value_pointer_net = PointerNet(args.hidden_size, args.col_embed_size, attention_type=args.column_att)

        # initial the embedding layers
        nn.init.xavier_normal_(self.production_embed.weight.data)
        nn.init.xavier_normal_(self.type_embed.weight.data)
        nn.init.xavier_normal_(self.N_embed.weight.data)
        print('Use Column Pointer: ', True if self.use_column_pointer else False)

    def forward(self, examples):
        args = self.args
        # now should implement the examples
        # "grammar" is the SemQL language. It contains lookup tables (string to id <--> id to string)
        batch = Batch(examples, self.grammar, cuda=self.args.cuda)

        table_appear_mask = batch.table_appear_mask

        # We use our transformer encoder to encode question together with the schema (columns and tables). See "TransformerEncoder" for details
        question_encodings, column_encodings, table_encodings, value_encodings, transformer_pooling_output = self.encoder(batch.src_sents,
                                                                                                                          batch.table_sents,
                                                                                                                          batch.table_names,
                                                                                                                          batch.values)
        question_encodings = self.dropout(question_encodings)

        # Source encodings to create the sketch (the AST without the leaf-nodes)
        utterance_encodings_sketch_linear = self.att_sketch_linear(question_encodings)
        # Source encodings to create the leaf-nodes
        utterance_encodings_lf_linear = self.att_lf_linear(question_encodings)

        dec_init_vec = self.init_decoder_state(transformer_pooling_output)
        h_tm1 = dec_init_vec
        action_probs = [[] for _ in examples]

        zero_action_embed = Variable(self.new_tensor(args.action_embed_size).zero_())
        zero_type_embed = Variable(self.new_tensor(args.type_embed_size).zero_())

        sketch_attention_history = list()

        ####################### PART 1: figuring out the sketch-action-loss ###########################################
        for t in range(batch.max_sketch_num):
            # This is the case if it the first decoding step. We initialize the inputs with zero
            # (while the last cell state is initialized from the last state of the decoder, see h_tm1).
            # This makes sense if you think about it: into the RNN we only feed information from the previous step.
            # In the first step we don't have this.
            if t == 0:
                x = Variable(self.new_tensor(len(batch), self.sketch_decoder_lstm.input_size).zero_(),
                             requires_grad=False)
            # if it is not the first step, we need to set together the information from the last step we wanna input
            # to the LSTM. This is all based on the Chapter 2.3 in TranX (first equation of the decoder):
            # the input is set together by the last action which is here the action embedding from
            # the "production_embed" array. The second attentional vector ~s (which is here "att_tm1") and the parent
            # feeding (the parent frontier field) p, which is here the "type_embed" embeddings.
            # All this information together form the input ("x") for the next step.

            else:
                a_tm1_embeds = []
                pre_types = []
                for e_id, example in enumerate(examples):

                    if t < len(example.sketch):
                        # we get the last action directly from the ground truth. That way we make sure we always feed the right action embedding (or production rule embedding to be exact) into the next RNN step.
                        action_tm1 = example.sketch[t - 1]
                        if type(action_tm1) in [semQL.Root1,
                                                semQL.Root,
                                                semQL.Sel,
                                                semQL.Filter,
                                                semQL.Sup,
                                                semQL.N,
                                                semQL.Order]:
                            a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]  # That way we get the next production rule embedding. A simple lookup.
                        else:
                            print(action_tm1, 'only for sketch')
                            quit() # The "example.sketch" should only contain sketch-actions (no leaf-actions). So if we reach this code it is an error scenario.
                            a_tm1_embed = zero_action_embed
                            pass
                    else:
                        a_tm1_embed = zero_action_embed     # add a "noop" action if this sample is shorter
                    # get the embeddings for the last action for the whole batch. Use Zero-Embeddings if no action.
                    a_tm1_embeds.append(a_tm1_embed)

                a_tm1_embeds = torch.stack(a_tm1_embeds)
                inputs = [a_tm1_embeds]

                for e_id, example in enumerate(examples):
                    if t < len(example.sketch):
                        action_tm = example.sketch[t - 1]
                        pre_type = self.type_embed.weight[self.grammar.type2id[type(action_tm)]]
                    else:
                        pre_type = zero_type_embed
                    pre_types.append(pre_type)

                pre_types = torch.stack(pre_types)

                inputs.append(att_tm1)
                inputs.append(pre_types)
                # The "inputs"-array (and so "x") contains three elements, exactly as described in the TranX-Paper (Decoder-Equation):
                # 1. ("a_tm1_embeds"): the action-embedding of the previous step
                # 2. ("att_tm1"): the attentional vector ~s of the previous step
                # 3. ("pre_types"): the parent feeding (see TranX-paper)
                x = torch.cat(inputs, dim=-1)

            src_mask = batch.src_token_mask

            # in here we do get the next step of the sketch_decoder_lstm, together with an attention mechanism, as described in TranX, 2.3
            # we only use (h_t, cell_t) only for the next step, to predict the sketch we use only att_t (keep in mind that h_t has already been used to calculate att_t)
            (h_t, cell_t), att_t, aw = self.step(x, h_tm1, question_encodings,
                                                 utterance_encodings_sketch_linear, self.sketch_decoder_lstm,
                                                 self.sketch_att_vec_linear,
                                                 src_token_mask=src_mask, return_att_weight=True)

            # This doesn't really seem to have an effect...
            sketch_attention_history.append(att_t)

            # get the Root possibility
            # Ursin: What do we afterwards have in apply_rule_prob? We have a vector size (batch * 46) with
            # probabilities. 46 is the amount of possible actions (see self.grammar.prod2id).
            # So we have a softmax over all possible actions for each example in the batch.

            # for details about "self.production_readout" go to the definition of it.
            apply_rule_prob = F.softmax(self.production_readout(att_t), dim=-1)

            # iterate over the batch.
            for e_id, example in enumerate(examples):
                # t is the current step in the whole "batch.max_sketch_num" loop. So we only add more actions if the current example is not smaller than this step.
                # This way we can handle the whole batch and just stop doing it for the small trees. Question: how do we teach the model to stop, if it never has to learn it? --> the stop is given by the grammar. If we predict e.g. an N action, we automatically know this part of the tree is over. so we just need to teach the model when to predict an N-action.
                if t < len(example.sketch):
                    action_t = example.sketch[t]
                    # We take the ground-truth action (action_t) and get the predicted probability for this action from
                    # the action-probability vector "apply_rule_prob". Be aware, e_id is only the current example.
                    act_prob_t_i = apply_rule_prob[e_id, self.grammar.prod2id[action_t.production]]
                    # in "action_probs" we store the sequence of prediction actions (for each example)
                    action_probs[e_id].append(act_prob_t_i)

            h_tm1 = (h_t, cell_t)
            att_tm1 = att_t

        # Note: the following line is quite important. It is basically the first part of the return-value of this whole function,
        # which is the sketch-loss. With the log()-function we find out, how large the loss is (remember: if probability = 1.0, log(1.0) = 0 --> no loss. log(0.001) = -3 --> large loss)
        # we then simply sum the loss up for all sketch-actions in an example, so we have a simple array of 64 (batch size) sketch losses.
        # Technically:
        # we do a sum(log()) over all action-probabilities of a sample (the loop is only to get over all samples of a batch).
        sketch_prob_var = torch.stack(
            [torch.stack(action_probs_i, dim=0).log().sum() for action_probs_i in action_probs], dim=0)

        ####################### PART 2: Create Schema (Column & Table) Embeddings ###########################################
        # What we see here in the next few lines is actually the schema encoder as described in IRNet
        # 2.3 "Schema Encoder".
        # IMPORTANT: by using the transformer encoder, this here simplifies quite a bit!

        # here we just create a tensor from "col_hot_type". Keep in mind: the col_hot_type is the type of matching ("exact" vs. "partial"). It basically states how well
        # a word matched with a column.
        col_type = self.input_type(batch.col_hot_type)

        # we create a linear layer around the col_type tensor.
        col_type_var = self.col_type(col_type)

        # We then also add an additional vector for the column type (the "phi" in the third equation of the schema encoder)
        table_embedding = column_encodings + col_type_var

        schema_embedding = table_encodings

        value_embedding = value_encodings

        batch_table_dict = batch.col_table_dict
        table_enable = np.zeros(shape=(len(examples)))
        action_probs = [[] for _ in examples]

        h_tm1 = dec_init_vec

        ####################### PART 3: figuring out the leaf-action ###########################################

        # important to understand: while we here still work with all actions (remember, the t-1 action can be anything!), we
        # are in the end only interested in the actions creating a Leaf-Node (so A, C and T). We therefore also work with "example.tgt_actions" and not with "example.sketch" as above.
        # in the end, when creating the loss, we only have a look at this three action types.
        for t in range(batch.max_action_num):
            if t == 0:
                # x = self.lf_begin_vec.unsqueeze(0).repeat(len(batch), 1)
                x = Variable(self.new_tensor(len(batch), self.lf_decoder_lstm.input_size).zero_(), requires_grad=False)
            else:
                a_tm1_embeds = []
                pre_types = []

                # it is very important to understand that this loop only selecting the right embedding based on the action in the last step!
                # So if the last action (we use the ground truth here) was a sketch-action, we simply select the embedding of its production rule.
                # If it was though a A or C action, we choose the depending column/table embedding.
                # The embedding is then part of the input in the next decoder step, as we see in TranX 2.3 (Decoder). It is not used to compare any loss, as this has already been done in the last step!
                for e_id, example in enumerate(examples):
                    if t < len(example.tgt_actions):
                        action_tm1 = example.tgt_actions[t - 1]
                        # We still need all the "Sketch-Action" types as they could be the t-1 action, before creating a leaf-node.
                        if type(action_tm1) in [semQL.Root1,
                                                semQL.Root,
                                                semQL.Sel,
                                                semQL.Filter,
                                                semQL.Sup,
                                                semQL.N,
                                                semQL.Order,
                                                ]:

                            a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]     # for sketch actions, we can just feed in the action (or exact: the production rule) embedding.
                        else:
                            # previous action C is a leaf-node, so we wanna feed in the right Column-embedding. We select the column idx by using the ground truth "id_c".
                            if isinstance(action_tm1, semQL.C):
                                a_tm1_embed = self.column_rnn_input(table_embedding[e_id, action_tm1.id_c])
                            # previous action T is a leaf-node, so we wanna feed in the right Table-embedding. We select the table by using the ground truth "id_c".
                            elif isinstance(action_tm1, semQL.T):
                                a_tm1_embed = self.table_rnn_input(schema_embedding[e_id, action_tm1.id_c])
                            # previous action V is a leaf-node, so we wanna feed in the right Value-embedding. We select the Value by using the ground truth "id_c".
                            elif isinstance(action_tm1, semQL.V):
                                a_tm1_embed = self.value_rnn_input(value_embedding[e_id, action_tm1.id_c])
                            # action A is handled like a normal sketch-action.
                            elif isinstance(action_tm1, semQL.A):
                                a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                            else:
                                print(action_tm1, 'not implement')
                                quit()
                                a_tm1_embed = zero_action_embed
                                pass

                    else:
                        a_tm1_embed = zero_action_embed
                    a_tm1_embeds.append(a_tm1_embed)

                a_tm1_embeds = torch.stack(a_tm1_embeds)

                inputs = [a_tm1_embeds]

                # very similar to the part above, but we consider here the "tgt_actions" to create the parent-feeding.
                # tgt t-1 action type
                for e_id, example in enumerate(examples):
                    if t < len(example.tgt_actions):
                        action_tm = example.tgt_actions[t - 1]
                        pre_type = self.type_embed.weight[self.grammar.type2id[type(action_tm)]]
                    else:
                        pre_type = zero_type_embed
                    pre_types.append(pre_type)

                pre_types = torch.stack(pre_types)

                inputs.append(att_tm1)

                inputs.append(pre_types)

                x = torch.cat(inputs, dim=-1)

            src_mask = batch.src_token_mask

            # we use a second RNN to predict the next actions for the leaf-nodes. Everything else stays the same as above
            (h_t, cell_t), att_t, aw = self.step(x, h_tm1, question_encodings,
                                                 utterance_encodings_lf_linear, self.lf_decoder_lstm,
                                                 self.lf_att_vec_linear,
                                                 src_token_mask=src_mask, return_att_weight=True)

            # the simple probability without the pointer-network is only needed for A actions. It is similar to the sketch action loss.
            apply_rule_prob = F.softmax(self.production_readout(att_t), dim=-1)

            ####################### PART 4: Selecting the right column/table/value ###########################################
            # We now want to calcuate the loss for selecting the right table/column/value. To do so, we use a point-network on top of the output of the decoder RNN.
            # Be aware that we don't calculate any loss for the sketch here, but only for the four leaf node actions C, T, A and V.

            table_appear_mask_val = torch.from_numpy(table_appear_mask)
            if self.cuda:
                table_appear_mask_val = table_appear_mask_val.cuda()

            # to my understanding the difference is not using pointer-networks or not, but using memory augmented pointer networks or just normal ones.
            if self.use_column_pointer:
                gate = torch.sigmoid(self.prob_att(att_t))
                # this equation can be found in the IRNet-Paper, at the end of chapter 2. See the comments in the paper.
                weights = self.column_pointer_net(src_encodings=table_embedding, query_vec=att_t.unsqueeze(0), src_token_mask=None) * table_appear_mask_val * gate + \
                          self.column_pointer_net(src_encodings=table_embedding, query_vec=att_t.unsqueeze(0), src_token_mask=None) * (1 - table_appear_mask_val) * (1 - gate)
            else:
                # remember: a pointer network basically just selecting a column from "table_embedding". It is a simplified attention mechanism
                weights = self.column_pointer_net(src_encodings=table_embedding, query_vec=att_t.unsqueeze(0), src_token_mask=batch.table_token_mask)

            # As not every question in the batch has the same number of columns, we need to mask out the unused columns before using the softmax.
            # The "masked_fill_" function fills every position with a "True"  with the given value (minus infinity).
            # So the remaining columns (the M in the beginning) is the actual columns.
            weights.data.masked_fill_(batch.table_token_mask.bool(), -float('inf'))

            # Calculate the probabilities for the selected columns.
            column_attention_weights = F.softmax(weights, dim=-1)

            # We do the same again to select a table.
            table_weights = self.table_pointer_net(src_encodings=schema_embedding, query_vec=att_t.unsqueeze(0),
                                                   src_token_mask=None)

            # The first part of masking the tables is basically the same as for the columns: not each example of the batch has the same
            # columns, therefore we need to mask out the unused ones.
            schema_token_mask = batch.schema_token_mask.expand_as(table_weights)
            table_weights.data.masked_fill_(schema_token_mask.bool(), -float('inf'))

            # the second part of the masking is more complex: based which column we chose on the last step, only one (or sometimes a few)
            # tables are possible - namely the ones which contain an attribute with this name. To implement this we save the last chosen column in "table_enable".
            # The "batch.table_dict_mask" is then a lookup table for the column <-> table relation (with the exception case for 0, where all tables are possible).
            # We then mask out all impossible column/table combinations by applying a second masking..
            table_dict = [batch_table_dict[x_id][int(x)] for x_id, x in enumerate(table_enable.tolist())]
            table_mask = batch.table_dict_mask(table_dict)
            table_weights.data.masked_fill_(table_mask.bool(), -float('inf'))

            # Calculate the probabilities for the selected columns.
            table_weights = F.softmax(table_weights, dim=-1)

            # Select a value with the pointer network
            value_weights = self.value_pointer_net(src_encodings=value_embedding, query_vec=att_t.unsqueeze(0),
                                                   src_token_mask=None)

            # As not every question in the batch has the same number of values, we need to mask out the unused values before using the softmax.
            # The "masked_fill_" function fills every position with a "True"  with the given value (minus infinity).
            # So the remaining columns (the M in the beginning) is the actual columns.
            # TODO: remember already "used" values and mask them out. We might also avoid masking if there is only one value per row.
            value_weights.data.masked_fill_(batch.value_token_mask.bool(), -99999)

            # Calculate the probabilities for the selected values.
            value_weights = F.softmax(value_weights, dim=-1)

            # Now we calculate the loss, but only for the leaf actions (A, C and T).
            # We are not interested in the loss of the sketch, as this has already been done in Part 1. The "action_probs" array will contain only entries for the leaf actions.
            for e_id, example in enumerate(examples):
                if t < len(example.tgt_actions):
                    action_t = example.tgt_actions[t]
                    if isinstance(action_t, semQL.C):
                        table_appear_mask[e_id, action_t.id_c] = 1  # not 100% sure, but to my understanding we use the column/table combinations for the memory-pointer network.
                        table_enable[e_id] = action_t.id_c  # make sure that in the next step, where we select a table, only existing column/table combinations appear. See masking above.
                        act_prob_t_i = column_attention_weights[e_id, action_t.id_c]  # ground truth says we are looking for a C action, we get the probability of predicting the right column (which is the value of the column-softmax at index id_c)
                        action_probs[e_id].append(act_prob_t_i)
                    elif isinstance(action_t, semQL.T):
                        act_prob_t_i = table_weights[e_id, action_t.id_c]  # ground truth says we are looking for a T action, we get the probability of predicting the right table (which is the value of the table-softmax at index id_c)
                        action_probs[e_id].append(act_prob_t_i)
                    elif isinstance(action_t, semQL.V):
                        act_prob_t_i = value_weights[e_id, action_t.id_c]  # ground truth says we are looking for a V action, we get the probability of predicting the right Value (which is the value of the value-softmax at index id_c)
                        action_probs[e_id].append(act_prob_t_i)
                    elif isinstance(action_t, semQL.A):     # action A is handled as a normal sketch action: we take the index of the ground-truth production rule and see with what a probability we would have predicted that.
                        act_prob_t_i = apply_rule_prob[e_id, self.grammar.prod2id[action_t.production]]
                        action_probs[e_id].append(act_prob_t_i)
                    else:
                        pass
            h_tm1 = (h_t, cell_t)
            att_tm1 = att_t

        # same as above for sketch_prob_var: we sum up the loss per sample.
        lf_prob_var = torch.stack(
            [torch.stack(action_probs_i, dim=0).log().sum() for action_probs_i in action_probs], dim=0)

        return [sketch_prob_var, lf_prob_var]

    def parse(self, examples, beam_size=5):
        """
        Method for prediction. Keep in mind that we handle only one example at the time here, but the first dimension is mostly the
        beam-size, so the n-possible hypothesis for this example!
        :param examples:
        :param beam_size:
        :return:
        """

        # Seems we use the same Batch class to keep the implementation similar to the training case
        batch = Batch([examples], self.grammar, cuda=self.args.cuda)

        # next lines is exactly the same as in the training case. Encode the source sentence.

        # We use our transformer encoder to encode question together with the schema (columns and tables). See "TransformerEncoder" for details
        question_encodings, column_encodings, table_encodings, value_encodings, transformer_pooling_output = self.encoder(batch.src_sents,
                                                                                                                          batch.table_sents,
                                                                                                                          batch.table_names,
                                                                                                                          batch.values)

        utterance_encodings_sketch_linear = self.att_sketch_linear(question_encodings)
        utterance_encodings_lf_linear = self.att_lf_linear(question_encodings)

        dec_init_vec = self.init_decoder_state(transformer_pooling_output)
        h_tm1 = dec_init_vec

        t = 0
        beams = [Beams(is_sketch=True)]  # we start with one initial Beam, from which we will create "beam_size" new beams in each round of the loop.
        completed_beams = []

        ####################### PART 1: creating the sketch ###########################################
        # in the following lines we create n completed sketch-beams by using the question-encoder and the decoder. In this part we do not use information about the schema, only about the question.
        # The beam-search allows us to only take the n-best candidates into the next decoder step.
        # We kow when a candidate is completed based on the grammar: if there is no more sketch action possible, we consider the sketch-candidate (or hypothesis) to be completed.
        # While we create n sketch candidates, we will only use the best of these to continue in Part 3, where we select the columns and tables for the leaf actions.

        # We either stop if we have done all beams or when we hit a maximum decoding steps (40 by default).
        while len(completed_beams) < beam_size and t < self.args.decode_max_time_step:
            hyp_num = len(beams)
            # we always keep n-beams in parallel - so we create here the data structure for it.
            exp_src_enconding = question_encodings.expand(hyp_num, question_encodings.size(1), question_encodings.size(2))
            exp_src_encodings_sketch_linear = utterance_encodings_sketch_linear.expand(hyp_num, utterance_encodings_sketch_linear.size(1),
                                                                                       utterance_encodings_sketch_linear.size(2))
            if t == 0:
                with torch.no_grad():
                    x = Variable(self.new_tensor(1, self.sketch_decoder_lstm.input_size).zero_())
            else:
                a_tm1_embeds = []
                pre_types = []
                for e_id, hyp in enumerate(beams):
                    action_tm1 = hyp.actions[-1]  # the last action we decoded
                    if type(action_tm1) in [semQL.Root1,  # if the last action was a sketch-action
                                            semQL.Root,
                                            semQL.Sel,
                                            semQL.Filter,
                                            semQL.Sup,
                                            semQL.N,
                                            semQL.Order]:
                        a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]  # get the embedding for the last action (exact action - so the production rule)
                    else:
                        raise ValueError('unknown action %s' % action_tm1)

                    a_tm1_embeds.append(a_tm1_embed)
                a_tm1_embeds = torch.stack(a_tm1_embeds)
                inputs = [a_tm1_embeds]  # so in here we have the embeddings for the last actions of each beam (only if it is a sketch action!)

                for e_id, hyp in enumerate(beams):
                    action_tm = hyp.actions[-1]
                    pre_type = self.type_embed.weight[self.grammar.type2id[type(action_tm)]]  # here we add in addition the type embedding of the last action ("parent feeding", see the TranX-paper)
                    pre_types.append(pre_type)

                pre_types = torch.stack(pre_types)

                inputs.append(att_tm1)
                inputs.append(pre_types)
                # The "inputs"-array (and so "x") contains three elements, exactly as described in the TranX-Paper (Decoder-Equation):
                # 1. ("a_tm1_embeds"): the action-embedding of the previous step
                # 2. ("att_tm1"): the attentional vector ~s of the previous step
                # 3. ("pre_types"): the parent feeding (see TranX-paper)
                x = torch.cat(inputs, dim=-1)

            # do the next decoder step - see TranX 2.3. Keep in mind that we don't really work with batches here (in opposite to the training), but we use the
            # first dimension for the beam-size. This means that h_tm1 always has a size of n-beams, and this are in every step representing the candidates which are still
            # "in the beam".
            (h_t, cell_t), att_t = self.step(x, h_tm1, exp_src_enconding,
                                             exp_src_encodings_sketch_linear, self.sketch_decoder_lstm,
                                             self.sketch_att_vec_linear,
                                             src_token_mask=None)

            # get the probabilities for the sketch actions. Not entirely sure why log-softmax and not normal softmax as above (effect: a value close to 0 will become somewhere between [-3, -4]
            apply_rule_log_prob = F.log_softmax(self.production_readout(att_t), dim=-1)

            new_hyp_meta = []  # this list is filled over the next few lines: it contains information about possible next production rules for each beam.
            for hyp_id, hyp in enumerate(beams):
                # Get possible next action class (only one - which makes sense if you look at the syntax, as there is always only one possible next class - but obviously many possible production rules!).
                # This is based on the actions in the beam until now (so e.g. a "Root1" can only be followed by "Root")
                action_class = hyp.get_availableClass()
                if action_class in [semQL.Root1,
                                    semQL.Root,
                                    semQL.Sel,
                                    semQL.Filter,
                                    semQL.Sup,
                                    semQL.N,
                                    semQL.Order]:
                    possible_productions = self.grammar.get_production(action_class)  # a list of possible production rules for the action class.
                    for possible_production in possible_productions:
                        prod_id = self.grammar.prod2id[possible_production]  # map the production rule to it's id (we need the same id e.g. to look up the embeddings)
                        prod_score = apply_rule_log_prob[hyp_id, prod_id]  # how low/high is the probability for this production rule?
                        new_hyp_score = hyp.score + prod_score.data.cpu()  # how good is the whole beam with this new production rule added?
                        meta_entry = {'action_type': action_class, 'prod_id': prod_id,
                                      'score': prod_score, 'new_hyp_score': new_hyp_score,
                                      'prev_hyp_id': hyp_id}
                        new_hyp_meta.append(meta_entry)
                else:
                    raise RuntimeError('No right action class')

            # if there is no new possible production rules for all beams, we stop the whole loop.
            if not new_hyp_meta: break

            new_hyp_scores = torch.stack([x['new_hyp_score'] for x in new_hyp_meta], dim=0)
            # return the k best beam-candidates. k is determined either by how many "open" beams we still have to fullfill, or by how many possible production rules candidates
            # we have (so if we have only 1 possible candidate, but still 2 open beams, we can still only use the one candidate)
            top_new_hyp_scores, meta_ids = torch.topk(new_hyp_scores, k=min(new_hyp_scores.size(0), beam_size - len(completed_beams)))
            # second return value, meta_ids, is referencing to the index of the k-candidates.

            live_hyp_ids = []
            new_beams = []

            # in this loop we basically just create new beams from the old ones, based on the top-k-candidates we selected above.
            for new_hyp_score, meta_id in zip(top_new_hyp_scores.data.cpu(), meta_ids.data.cpu()):
                action_info = ActionInfo()
                hyp_meta_entry = new_hyp_meta[meta_id]
                prev_hyp_id = hyp_meta_entry['prev_hyp_id']
                prev_hyp = beams[prev_hyp_id]  # prev_hyp is referring to the "old" beam
                action_type_str = hyp_meta_entry['action_type']
                prod_id = hyp_meta_entry['prod_id']
                if prod_id < len(self.grammar.id2prod):
                    production = self.grammar.id2prod[prod_id]
                    action = action_type_str(list(action_type_str._init_grammar()).index(production))   # here we create a new action with the id_c of the production rule as constructor argument
                else:
                    raise NotImplementedError

                action_info.action = action
                action_info.t = t  # remember, t represents the current decoding-step (e.g. first step = 0)
                action_info.score = hyp_meta_entry['score']
                new_hyp = prev_hyp.clone_and_apply_action_info(action_info)  # create a new beam object from the old one by applying the production rule we chose
                new_hyp.score = new_hyp_score  # the new total score of this beam
                new_hyp.inputs.extend(prev_hyp.inputs)

                # not entirely sure how a beam can not be valid at this point... check implementation details at one point.
                if new_hyp.is_valid is False:
                    continue

                # a completed beam means there is no next action class. This happens e.g. if there are only N's left at the end of each tree-leaf. See implementation of Beam.get_availableClass() and Action.get_next_action()
                if new_hyp.completed:
                    completed_beams.append(new_hyp)
                else:
                    new_beams.append(new_hyp)
                    live_hyp_ids.append(prev_hyp_id)  # this list is referring to the beams we keep "alive", so the ones we created the new beams (the old beam + one production rule) from.

            # this is what we take over to the next decoding step.
            if live_hyp_ids:
                # the next 2 lines are interesting: we don't need to remember the hidden-states/cell-states of the decoder for beams we anyway don't consider in the future steps.
                # so we basically select here only the beam-candidates that will "live" in the next steps! That's also why we keep track of the "prev_hyp_id".
                # it is further very possible that multiple "new beams" descend from the same parent beam. In that case, we just have this cell/hidden-state duplicated. This is by design.
                h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
                att_tm1 = att_t[live_hyp_ids]
                beams = new_beams  # we only continue with the new n-beams. This makes sense: they are anyway copies of the old one, but with the additional production rule.
                t += 1  # increase decoding step. Remember: this is necessary to avoid getting infinite large sketch (see decode_max_time_step)
            else:
                break  # this is the case if we completed all beams in the step.

        # now get the sketch result
        completed_beams.sort(key=lambda hyp: -hyp.score)  # we want to have the best beam on top.
        if len(completed_beams) == 0:
            return [[], []]

        # it's important to note that only the best beam is further used for Part 3 (and returned).
        # While we have another beam-search in Part 3, this is fully relying on the best sketch-beam selected here.
        sketch_actions = completed_beams[0].actions
        # sketch_actions = examples.sketch

        padding_sketch = self.padding_sketch(sketch_actions)

        ####################### PART 2: Create Schema (Column & Table) Embeddings ###########################################
        # this part is exacty the same as Part 2 in training. It is further independent of chosen sketch in Part 1
        col_type = self.input_type(batch.col_hot_type)

        col_type_var = self.col_type(col_type)

        table_embedding = column_encodings + col_type_var

        schema_embedding = table_encodings

        value_embedding = value_encodings

        batch_table_dict = batch.col_table_dict

        h_tm1 = dec_init_vec

        ####################### PART 3: Select columns and tables (and A) ###########################################
        # Based on the Sketch of Part 1 we now know how the AST looks. We even added "dummy" actions for C, T and A. But what is still missing,
        # is selecting the right index to the tables and columns (and production rules for A). This is what we do in Part 3.
        # We use the best sketch from Part 1, the schema encoding from Part 2 and run a new RNN to predict columns/tables.
        # it is important to understand that we run the whole decoding again, including the sketch part. This is due to the sequential nature of the RNN: we need to
        # bring it into the right state to predict the leaf nodes. It's important to note though that the sketch-actions are already pre-defined by the sketch from above
        # (with the "else"-case, which will always use the sketch action), so in those cases we just need to use the right action (or more exactly, production-rule) embedding to keep the
        # RNN going on.

        t = 0
        beams = [Beams(is_sketch=False)]
        completed_beams = []

        while len(completed_beams) < beam_size and t < self.args.decode_max_time_step:
            hyp_num = len(beams)

            # expand value. Similar to Part 1, but here we als expand the table/schema-embedding
            exp_src_encodings = question_encodings.expand(hyp_num, question_encodings.size(1), question_encodings.size(2))
            exp_utterance_encodings_lf_linear = utterance_encodings_lf_linear.expand(hyp_num, utterance_encodings_lf_linear.size(1),
                                                                                     utterance_encodings_lf_linear.size(2))
            exp_table_embedding = table_embedding.expand(hyp_num, table_embedding.size(1),
                                                         table_embedding.size(2))

            exp_schema_embedding = schema_embedding.expand(hyp_num, schema_embedding.size(1),
                                                           schema_embedding.size(2))

            # not a 100& sure what this block is for. Both, "table_appear_mask" and "table_enable" seem to be there to mark the tables which
            # are used if we predict a column, and we remember the index of the column in the mask.
            table_appear_mask = batch.table_appear_mask
            table_appear_mask = np.zeros((hyp_num, table_appear_mask.shape[1]), dtype=np.float32)
            table_enable = np.zeros(shape=(hyp_num))
            for e_id, hyp in enumerate(beams):
                for act in hyp.actions:
                    if type(act) == semQL.C:
                        table_appear_mask[e_id][act.id_c] = 1
                        table_enable[e_id] = act.id_c

            if t == 0:
                with torch.no_grad():
                    x = Variable(self.new_tensor(1, self.lf_decoder_lstm.input_size).zero_())
            else:
                a_tm1_embeds = []
                pre_types = []

                # it is very important to understand that this loop only selecting the right embedding based on the action we choose in the last step!
                # so if we selected a sketch-action (or A) in the last step, we simply select the embedding of this production rule. If we though selected an A or C action,
                # we choose the depending column/table embedding.
                # The embedding is then part of the input in the next decoder step, as we see in TranX 2.3 (Decoder).
                for e_id, hyp in enumerate(beams):
                    action_tm1 = hyp.actions[-1]
                    if type(action_tm1) in [semQL.Root1,
                                            semQL.Root,
                                            semQL.Sel,
                                            semQL.Filter,
                                            semQL.Sup,
                                            semQL.N,
                                            semQL.Order]:

                        a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                        hyp.sketch_step += 1
                    elif isinstance(action_tm1, semQL.C):
                        a_tm1_embed = self.column_rnn_input(table_embedding[0, action_tm1.id_c])    # the id_c is to select the right column-embedding, based on the column index we selected in the last step. The 0 is only necessary because the table_embedding has 2 dimensions (first is batch, which we don't use in inference).
                    elif isinstance(action_tm1, semQL.T):
                        a_tm1_embed = self.table_rnn_input(schema_embedding[0, action_tm1.id_c])
                    elif isinstance(action_tm1, semQL.V):
                        a_tm1_embed = self.value_rnn_input(value_embedding[0, action_tm1.id_c])
                    elif isinstance(action_tm1, semQL.A):
                        a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]     # A behaves similar to the sketch actions.
                    else:
                        raise ValueError('unknown action %s' % action_tm1)

                    a_tm1_embeds.append(a_tm1_embed)

                a_tm1_embeds = torch.stack(a_tm1_embeds)

                inputs = [a_tm1_embeds]

                for e_id, hyp in enumerate(beams):
                    action_tm = hyp.actions[-1]
                    pre_type = self.type_embed.weight[self.grammar.type2id[type(action_tm)]]
                    pre_types.append(pre_type)

                pre_types = torch.stack(pre_types)

                inputs.append(att_tm1)
                inputs.append(pre_types)
                x = torch.cat(inputs, dim=-1)

            # The "inputs"-array (and so "x") contains three elements, exactly as described in the TranX-Paper (Decoder-Equation):
            # 1. ("a_tm1_embeds"): the action-embedding of the previous step
            # 2. ("att_tm1"): the attentional vector ~s of the previous step
            # 3. ("pre_types"): the parent feeding (see TranX-paper)
            (h_t, cell_t), att_t = self.step(x, h_tm1, exp_src_encodings,
                                             exp_utterance_encodings_lf_linear, self.lf_decoder_lstm,
                                             self.lf_att_vec_linear,
                                             src_token_mask=None)

            # this probability we actually only use in case of the A action, to select the right production rule. For sketch rules we use the sketch from Part 1, for
            # C and T we use the results from the pointer network.
            apply_rule_log_prob = F.log_softmax(self.production_readout(att_t), dim=-1)

            table_appear_mask_val = torch.from_numpy(table_appear_mask)

            if self.args.cuda: table_appear_mask_val = table_appear_mask_val.cuda()

            # use the pointer network, similar to the training part.
            if self.use_column_pointer:
                gate = torch.sigmoid(self.prob_att(att_t))
                weights = self.column_pointer_net(src_encodings=exp_table_embedding, query_vec=att_t.unsqueeze(0),
                                                  src_token_mask=None) * table_appear_mask_val * gate + self.column_pointer_net(
                    src_encodings=exp_table_embedding, query_vec=att_t.unsqueeze(0),
                    src_token_mask=None) * (1 - table_appear_mask_val) * (1 - gate)
                # weights = weights + self.col_attention_out(exp_embedding_differ).squeeze()
            else:
                weights = self.column_pointer_net(src_encodings=exp_table_embedding, query_vec=att_t.unsqueeze(0),
                                                  src_token_mask=batch.table_token_mask)

            # TODO: should this mask not get activated? In my understanding it is only here because different examples in the batch have different lengths
            # TODO  ------------> maybe not necessary because we anyway one have one beam - so only one example.
            # weights.data.masked_fill_(exp_col_pred_mask, -float('inf'))

            # the probabilities we use in case of C action
            column_selection_log_prob = F.log_softmax(weights, dim=-1)

            table_weights = self.table_pointer_net(src_encodings=exp_schema_embedding, query_vec=att_t.unsqueeze(0),
                                                   src_token_mask=None)
            # table_weights = self.table_pointer_net(src_encodings=exp_schema_embedding, query_vec=att_t.unsqueeze(0), src_token_mask=None)

            schema_token_mask = batch.schema_token_mask.expand_as(table_weights)
            table_weights.data.masked_fill_(schema_token_mask.bool(), -float('inf'))

            table_dict = [batch_table_dict[0][int(x)] for x_id, x in enumerate(table_enable.tolist())]
            table_mask = batch.table_dict_mask(table_dict)
            table_weights.data.masked_fill_(table_mask.bool(), -float('inf'))

            # the probabilities we use in case of T action
            table_weights = F.log_softmax(table_weights, dim=-1)

            # Select a value with the pointer network
            value_weights = self.value_pointer_net(src_encodings=value_embedding, query_vec=att_t.unsqueeze(0),
                                                   src_token_mask=None)

            # As not every question in the batch has the same number of values, we need to mask out the unused values before using the softmax.
            # The "masked_fill_" function fills every position with a "True"  with the given value (minus infinity).
            # So the remaining columns (the M in the beginning) is the actual columns.
            # TODO: remember already "used" values and mask them out
            value_weights.data.masked_fill_(batch.value_token_mask.bool(), -float('inf'))

            new_hyp_meta = []
            for hyp_id, hyp in enumerate(beams):
                # TODO: should change this
                # This is a very important part, as here the sketch from Part 1 comes together with Part 3. Important is to remember that the sketch got padded with "dummy" A, C, T and V actions.
                # here, we create new candidates for each case, based on the probabilities we calculated above with the Softmax.
                if type(padding_sketch[t]) == semQL.A:
                    # for action A we use the same mechanism as in creating the sketch production rules. We already know it's action A, so we get all possible production rules.
                    possible_productions = self.grammar.get_production(semQL.A)
                    for possible_production in possible_productions:
                        prod_id = self.grammar.prod2id[possible_production]
                        prod_score = apply_rule_log_prob[hyp_id, prod_id]   # we look up the probability for each possible production rule

                        new_hyp_score = hyp.score + prod_score.data.cpu()   # we create a new score for this hypothesis. Remember, due to the log-softmax, low probabilities results in higher negative scores.
                        meta_entry = {'action_type': semQL.A, 'prod_id': prod_id,
                                      'score': prod_score, 'new_hyp_score': new_hyp_score,
                                      'prev_hyp_id': hyp_id}
                        new_hyp_meta.append(meta_entry)

                elif type(padding_sketch[t]) == semQL.C:
                    # the column (C), table (T) and Value (V) case is different: here we want to create candidates for each possible column.
                    for col_id, _ in enumerate(batch.table_sents[0]):
                        col_sel_score = column_selection_log_prob[hyp_id, col_id]   # the probability for each column we calculated before by using the schema encoding!
                        new_hyp_score = hyp.score + col_sel_score.data.cpu()
                        meta_entry = {'action_type': semQL.C, 'col_id': col_id,
                                      'score': col_sel_score, 'new_hyp_score': new_hyp_score,
                                      'prev_hyp_id': hyp_id}
                        new_hyp_meta.append(meta_entry)
                elif type(padding_sketch[t]) == semQL.T:
                    # similar to the C action
                    for t_id, _ in enumerate(batch.table_names[0]):
                        t_sel_score = table_weights[hyp_id, t_id]
                        new_hyp_score = hyp.score + t_sel_score.data.cpu()

                        meta_entry = {'action_type': semQL.T, 't_id': t_id,
                                      'score': t_sel_score, 'new_hyp_score': new_hyp_score,
                                      'prev_hyp_id': hyp_id}
                        new_hyp_meta.append(meta_entry)
                elif type(padding_sketch[t]) == semQL.V:
                    # similar to the C and T action
                    for value_idx, _ in enumerate(batch.values[0]):
                        val_sel_score = value_weights[hyp_id, value_idx]
                        new_hyp_score = hyp.score + val_sel_score.data.cpu()

                        meta_entry = {'action_type': semQL.V, 'val_id': value_idx,
                                      'score': val_sel_score, 'new_hyp_score': new_hyp_score,
                                      'prev_hyp_id': hyp_id}
                        new_hyp_meta.append(meta_entry)

                else:
                    # but what happens if the next Action in the sketch is a sketch Action (which is often the case, as only the padded leaf actions will be handled by the statements above)?
                    # we just create one candidate which has exactly the same production rule as the sketch action (therefore "padding_sketch[t].production")
                    prod_id = self.grammar.prod2id[padding_sketch[t].production]
                    new_hyp_score = hyp.score + torch.tensor(0.0)   # by using a score of 0 for this case we make sure this does not change our scoring scheme: we want to make sure the score is only relating A, C and T in this part of the code! Remember, the sketch is already set above!
                    meta_entry = {'action_type': type(padding_sketch[t]), 'prod_id': prod_id,
                                  'score': torch.tensor(0.0), 'new_hyp_score': new_hyp_score,
                                  'prev_hyp_id': hyp_id}
                    new_hyp_meta.append(meta_entry)

            if not new_hyp_meta: break

            new_hyp_scores = torch.stack([x['new_hyp_score'] for x in new_hyp_meta], dim=0)
            # as above, we keep the n best hypothesis with the highest score. Remember, in each step the score added is in best case 0, but rather somewhere around - 3 due to the log-softmax
            top_new_hyp_scores, meta_ids = torch.topk(new_hyp_scores, k=min(new_hyp_scores.size(0), beam_size - len(completed_beams)))

            live_hyp_ids = []
            new_beams = []

            # in this loop, similar to Part 1, we create new beams from the old ones, based on the top-k-candidates we selected above.
            for new_hyp_score, meta_id in zip(top_new_hyp_scores.data.cpu(), meta_ids.data.cpu()):
                action_info = ActionInfo()
                hyp_meta_entry = new_hyp_meta[meta_id]
                prev_hyp_id = hyp_meta_entry['prev_hyp_id']
                prev_hyp = beams[prev_hyp_id]

                action_type_str = hyp_meta_entry['action_type']

                # a prod_id exist only if it is an A action or a sketch action
                if 'prod_id' in hyp_meta_entry:
                    prod_id = hyp_meta_entry['prod_id']

                if action_type_str == semQL.C:
                    col_id = hyp_meta_entry['col_id']
                    action = semQL.C(col_id)    # remember, in action C the id_c is the index of the column.
                elif action_type_str == semQL.T:
                    t_id = hyp_meta_entry['t_id']
                    action = semQL.T(t_id)  # and for action T, it is the index of the table.
                elif action_type_str == semQL.V:
                    val_id = hyp_meta_entry['val_id']
                    action = semQL.V(val_id)  # and for action V, it is the index of the Value.
                # This is the case for A actions or sketch actions.
                elif prod_id < len(self.grammar.id2prod):
                    production = self.grammar.id2prod[prod_id]
                    action = action_type_str(list(action_type_str._init_grammar()).index(production))   # here we create a new action with the id_c of the production rule as constructor argument
                else:
                    raise NotImplementedError

                action_info.action = action
                action_info.t = t
                action_info.score = hyp_meta_entry['score']

                new_hyp = prev_hyp.clone_and_apply_action_info(action_info)
                new_hyp.score = new_hyp_score
                new_hyp.inputs.extend(prev_hyp.inputs)

                if new_hyp.is_valid is False:
                    continue

                # The way a hypothesis is completed doesn't really change compared to Part 1. As soon as as there is no more Action class available, the hypothesis is completed.
                # what changes is the is_sketch-flag: the hypothesis will now only be done when also non-sketch actions (A, T and C) are set.
                if new_hyp.completed:
                    completed_beams.append(new_hyp)
                else:
                    new_beams.append(new_hyp)
                    live_hyp_ids.append(prev_hyp_id)

            # this is what we take over to the next decoding step. Exact similar to Part 1.
            if live_hyp_ids:
                h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
                att_tm1 = att_t[live_hyp_ids]

                beams = new_beams
                t += 1
            else:
                break

        # the completed beams are sorted, best (highest score) first.
        completed_beams.sort(key=lambda hyp: -hyp.score)

        # so what do we return in the end? We return the best sketch beam (sketch_actions) and the best full-beam (completed_beam).
        # The "completed_beam" also contains A, C and T actions and more important, in the C and T actions the index to the columns/tables!
        return [completed_beams, sketch_actions]

    def step(self, x, h_tm1, src_encodings, src_encodings_att_linear, decoder, attention_func, src_token_mask=None,
             return_att_weight=False):
        # h_t: (batch_size, hidden_size)
        h_t, cell_t = decoder(x, h_tm1)

        # calculate attention. See "dot_prod_attention" for more details.
        ctx_t, alpha_t = nn_utils.dot_prod_attention(h_t,
                                                     src_encodings, src_encodings_att_linear,
                                                     mask=src_token_mask)

        # we concat the hidden state and the context vector as input. Forget about the attention_function.
        # This is exactly as the described in TranX, 2.3 (equation with tanh)
        # this is just a linear function
        att_t = torch.tanh(attention_func(torch.cat([h_t, ctx_t], 1)))
        att_t = self.dropout(att_t)

        if return_att_weight:
            return (h_t, cell_t), att_t, alpha_t
        else:
            return (h_t, cell_t), att_t

    def init_decoder_state(self, enc_last_cell):
        h_0 = self.decoder_cell_init(enc_last_cell)
        h_0 = torch.tanh(h_0)

        return h_0, Variable(self.new_tensor(h_0.size()).zero_())

