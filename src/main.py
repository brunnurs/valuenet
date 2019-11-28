import datetime
import os

from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm

from src.config import read_arguments_train, write_config_to_file
from src.data_loader import get_data_loader
from src.encoder import get_encoder_model
from src.evaluation import evaluate
from src.optimizer import build_optimizer_encoder
from src.spider import spider_utils
from src.training import train

from src.utils import setup_device, set_seed_everywhere


def create_experiment_folder(model_output_dir, data_dir):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    experiment_name = "{}__{}".format(data_dir.upper(), timestamp)

    output_path = os.path.join(model_output_dir, experiment_name)
    os.makedirs(output_path, exist_ok=True)

    return experiment_name


if __name__ == '__main__':
    args = read_arguments_train()
    experiment_name = create_experiment_folder(args.model_output_dir, args.data_dir)
    write_config_to_file(args, args.model_output_dir, experiment_name)

    device, n_gpu = setup_device()
    set_seed_everywhere(args.seed, n_gpu)

    sql_data, table_data, val_sql_data, val_table_data = spider_utils.load_dataset(args.data_dir, use_small=args.toy)
    train_loader, dev_loader = get_data_loader(sql_data, val_sql_data, args.batch_size_encoder, True, False)

    model, tokenizer = get_encoder_model(args.encoder_pretrained_model)
    # TODO: build decoder model here

    num_train_steps = len(train_loader) * args.num_epochs
    optimizer, scheduler = build_optimizer_encoder(model,
                                                   num_train_steps,
                                                   args.learning_rate,
                                                   args.adam_eps,
                                                   args.warmup_steps,
                                                   args.weight_decay)

    tb_writer = SummaryWriter(os.path.join(args.model_output_dir, experiment_name))
    global_step = 0

    print("Start training with {} epochs".format(args.num_epochs))
    for epoch in tqdm(range(int(args.num_epochs))):
        train(device, train_loader, model, tokenizer, optimizer, scheduler, args.max_seq_length)
        evaluate()