import datetime
import os

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import read_arguments_train, write_config_to_file
from src.data_loader import get_data_loader
from src.encoder import get_encoder_model
from src.evaluation import evaluate
from src.optimizer import build_optimizer_encoder
from src.spider import spider_utils
from src.training import train
from sklearn.model_selection import train_test_split

from src.utils import setup_device, set_seed_everywhere, create_labels_for_dummy_task


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

    data_1, _, data_2, _ = spider_utils.load_dataset(args.data_dir, use_small=args.toy)
    data_total = data_1 + data_2

    train_data, test_data = train_test_split(data_total, test_size=0.2)

    train_loader, dev_loader = get_data_loader(train_data, test_data, args.batch_size_encoder, True, False)

    # TODO this might only be temporary for the dummy training task
    label_map = create_labels_for_dummy_task(data_1, data_2)

    model, tokenizer = get_encoder_model(args.encoder_pretrained_model, len(label_map))
    model.to(device)
    # TODO: build decoder model here

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
    for epoch in tqdm(range(int(args.num_epochs))):

        global_step = train(global_step,
                            tb_writer,
                            device,
                            train_loader,
                            model,
                            tokenizer,
                            optimizer,
                            scheduler,
                            label_map,
                            args.max_seq_length)

        print("Evaluate on the dev-set")
        eval_results = evaluate(model,
                                device,
                                tokenizer,
                                dev_loader,
                                epoch,
                                label_map,
                                output_path,
                                args.max_seq_length)

        for key, value in eval_results.items():
            tb_writer.add_scalar(key, value, global_step)

    tb_writer.close()
