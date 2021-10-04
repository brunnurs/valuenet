import argparse
import json
import os


class Config:
    DATA_PREFIX = "data"
    EXPERIMENT_PREFIX = "experiments"


def write_config_to_file(args, output_path):
    config_path = os.path.join(output_path, "args.json")

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(args.__dict__, f, indent=2)


def _add_model_configuration(parser):

    parser.add_argument('--cuda', default=True, action='store_true')

    # language model configuration
    parser.add_argument('--encoder_pretrained_model', default='facebook/bart-base', type=str)
    parser.add_argument('--max_seq_length', default=1024, type=int)

    # model configuration
    parser.add_argument('--column_pointer', action='store_true', default=True)
    parser.add_argument('--embed_size', default=300, type=int, help='size of word embeddings')
    parser.add_argument('--hidden_size', default=300, type=int, help='size of LSTM hidden states')
    parser.add_argument('--action_embed_size', default=128, type=int, help='size of word embeddings')
    parser.add_argument('--att_vec_size', default=300, type=int, help='size of attentional vector')
    parser.add_argument('--type_embed_size', default=128, type=int, help='size of word embeddings')
    parser.add_argument('--col_embed_size', default=300, type=int, help='size of word embeddings')
    parser.add_argument('--readout', default='identity', choices=['identity', 'non_linear'])
    parser.add_argument('--column_att', choices=['dot_prod', 'affine'], default='affine')
    parser.add_argument('--dropout', default=0.3, type=float, help='dropout rate')


def _add_postgresql_configuration(parser):
    parser.add_argument('--database_host', default='localhost', type=str)
    parser.add_argument('--database_port', default='18001', type=str)
    parser.add_argument('--database_user', default='postgres', type=str)
    parser.add_argument('--database_password', default='dummy', type=str)
    parser.add_argument('--database_schema', default='unics_cordis', type=str)


def read_arguments_train():
    parser = argparse.ArgumentParser(description="Run training with following arguments")

    # model configuration
    _add_model_configuration(parser)

    # general configuration
    parser.add_argument('--exp_name', default='exp', type=str)
    parser.add_argument('--seed', default=90, type=int)
    parser.add_argument('--toy', default=False, action='store_true')
    parser.add_argument('--data_set', default='spider', type=str)

    # training & optimizer configuration
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_epochs', default=5.0, type=float)

    parser.add_argument('--lr_base', default=1e-3, type=float)
    parser.add_argument('--lr_connection', default=1e-4, type=float)
    parser.add_argument('--lr_transformer', default=2e-5, type=float)
    # parser.add_argument('--adam_eps', default=1e-8, type=float)
    parser.add_argument('--scheduler_gamma', default=0.5, type=int)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--clip_grad', default=5., type=float)
    parser.add_argument('--loss_epoch_threshold', default=50, type=int)
    parser.add_argument('--sketch_loss_weight', default=1.0, type=float)

    parser.add_argument('--run_spider_evaluation_after_epoch', action='store_true', default=False,
                        help='Run evaluation on the spider test suite after each epoch. If false, only accuracy/sketch_accuracy on the SemQL result is calculated.')

    # prediction configuration (run after each epoch)
    parser.add_argument('--beam_size', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--decode_max_time_step', default=40, type=int,
                        help='maximum number of time steps used in decoding and sampling')

    args = parser.parse_args()

    args.data_dir = os.path.join(Config.DATA_PREFIX, args.data_set)
    args.model_output_dir = Config.EXPERIMENT_PREFIX

    print("*** parsed configuration from command line and combine with constants ***")

    for argument in vars(args):
        print("argument: {}={}".format(argument, getattr(args, argument)))

    return args


def read_arguments_evaluation():
    parser = argparse.ArgumentParser(description="Run evaluation with following arguments")

    # model configuration
    _add_model_configuration(parser)

    # evaluation
    parser.add_argument('--evaluation_type', default='spider', type=str)

    parser.add_argument('--model_to_load', type=str)
    parser.add_argument('--prediction_dir', type=str)
    parser.add_argument('--batch_size', default=1, type=int)

    # general configuration
    parser.add_argument('--seed', default=90, type=int)
    parser.add_argument('--data_set', default='spider', type=str)

    # prediction configuration
    parser.add_argument('--beam_size', default=1, type=int, help='beam size for beam search')
    parser.add_argument('--decode_max_time_step', default=40, type=int,
                        help='maximum number of time steps used in decoding and sampling')

    # DB config is only needed in case evaluation is executed on PostgreSQL DB
    _add_postgresql_configuration(parser)

    parser.add_argument('--database', default='cordis_temporary', type=str)

    args = parser.parse_args()

    args.data_dir = os.path.join(Config.DATA_PREFIX, args.data_set)

    print("*** parsed configuration from command line and combine with constants ***")

    for argument in vars(args):
        print("argument: {}={}".format(argument, getattr(args, argument)))

    return args


def read_arguments_manual_inference():
    parser = argparse.ArgumentParser(description="Run manual inference with following arguments")

    # model configuration
    _add_model_configuration(parser)

    # manual_inference
    parser.add_argument('--model_to_load', type=str)
    parser.add_argument('--api_key', default='1234', type=str)
    parser.add_argument('--ner_api_secret', default='PLEASE_ADD_YOUR_OWN_GOOGLE_API_KEY_HERE', type=str)

    # database configuration (in case of PostgreSQL, not needed for sqlite)
    _add_postgresql_configuration(parser)

    # general configuration
    parser.add_argument('--seed', default=90, type=int)
    parser.add_argument('--batch_size', default=1, type=int)

    # prediction configuration
    parser.add_argument('--beam_size', default=1, type=int, help='beam size for beam search')
    parser.add_argument('--decode_max_time_step', default=40, type=int,
                        help='maximum number of time steps used in decoding and sampling')

    args = parser.parse_args()

    print("*** parsed configuration from command line and combine with constants ***")

    for argument in vars(args):
        print("argument: {}={}".format(argument, getattr(args, argument)))

    return args
