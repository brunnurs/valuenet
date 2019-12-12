import datetime
import os

from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src import utils
from src.config import read_arguments_train, write_config_to_file
from src.data_loader import get_data_loader
from src.encoder import get_encoder_model
from src.evaluation import evaluate
from src.intermediate_representation import semQL
from src.model.model import IRNet
from src.optimizer import build_optimizer_encoder
from src.spider import spider_utils
from src.training import train

from src.utils import setup_device, set_seed_everywhere
from pytictoc import TicToc


def create_experiment_folder(model_output_dir):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    exp = "{}__{}".format("Spider", timestamp)

    out_path = os.path.join(model_output_dir, exp)
    os.makedirs(out_path, exist_ok=True)

    return exp, out_path


if __name__ == '__main__':
    args = read_arguments_train()
    experiment_name, output_path = create_experiment_folder(args.model_output_dir)
    write_config_to_file(args, args.model_output_dir, experiment_name)

    device, n_gpu = setup_device()
    set_seed_everywhere(args.seed, n_gpu)

    sql_data, table_data, val_sql_data, val_table_data = spider_utils.load_dataset(args.data_dir, use_small=args.toy)
    train_loader, dev_loader = get_data_loader(sql_data, val_sql_data, args.batch_size, True, False)

    grammar = semQL.Grammar()
    model = IRNet(args, grammar)
    model.to(device)

    model.word_emb = utils.load_word_emb(args.glove_embed_path)

    num_train_steps = len(train_loader) * args.num_epochs
    optimizer, scheduler = build_optimizer_encoder(model,
                                                   num_train_steps,
                                                   args.learning_rate,
                                                   args.adam_eps,
                                                   args.warmup_steps,
                                                   args.weight_decay)

    tb_writer = SummaryWriter(output_path)
    global_step = 0

    print("Start training with {} epochs".format(args.num_epochs))
    t = TicToc()
    for epoch in tqdm(range(int(args.num_epochs))):

        sketch_loss_weight = 1 if epoch < args.loss_epoch_threshold else args.sketch_loss_weight

        t.tic()
        global_step = train(global_step,
                            tb_writer,
                            train_loader,
                            table_data,
                            model,
                            optimizer,
                            scheduler,
                            args.clip_grad,
                            sketch_loss_weight=sketch_loss_weight)

        train_time = t.tocvalue()

        tqdm.write("Training of epoch {} finished after {} seconds. Evaluate on the dev-set".format(epoch, train_time))
        # eval_results = evaluate(model,
        #                         device,
        #                         tokenizer,
        #                         dev_loader,
        #                         epoch,
        #                         label_map,
        #                         output_path,
        #                         args.max_seq_length)

        # for key, value in eval_results.items():
        #     tb_writer.add_scalar(key, value, global_step)

    tb_writer.close()
