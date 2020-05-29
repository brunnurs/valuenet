import json
import os

import torch
from pytictoc import TicToc
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.model import IRNet
from config import read_arguments_train, write_config_to_file
from data_loader import get_data_loader
from evaluation import evaluate, transform_to_sql_and_evaluate_with_spider
from intermediate_representation import semQL
from optimizer import build_optimizer_encoder
from spider import spider_utils
from training import train
from utils import setup_device, set_seed_everywhere, save_model, create_experiment_folder

# initialize experiment tracking @ Weights & Biases
import wandb
wandb.init(project="proton")


if __name__ == '__main__':
    args = read_arguments_train()

    # log hyperparameters to Weights & Biases
    wandb.config.update(args)

    experiment_name, output_path = create_experiment_folder(args.model_output_dir, args.exp_name)
    print("Run experiment '{}'".format(experiment_name))

    write_config_to_file(args, output_path)

    device, n_gpu = setup_device()
    set_seed_everywhere(args.seed, n_gpu)

    sql_data, table_data, val_sql_data, val_table_data = spider_utils.load_dataset(args.data_dir, use_small=args.toy)
    train_loader, dev_loader = get_data_loader(sql_data, val_sql_data, args.batch_size, True, False)

    grammar = semQL.Grammar()
    model = IRNet(args, device, grammar)
    model.to(device)

    # track the model
    wandb.watch(model, log='parameters')

    num_train_steps = len(train_loader) * args.num_epochs
    optimizer, scheduler = build_optimizer_encoder(model,
                                                   num_train_steps,
                                                   args.lr_transformer, args.lr_connection, args.lr_base,
                                                   args.scheduler_gamma)

    tb_writer = SummaryWriter(output_path)
    global_step = 0
    best_acc = 0.0

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
                            args.clip_grad,
                            sketch_loss_weight=sketch_loss_weight)

        train_time = t.tocvalue()

        tqdm.write("Training of epoch {0} finished after {1:.2f} seconds. Evaluate now on the dev-set".format(epoch, train_time))
        with torch.no_grad():
            sketch_acc, acc, _, predictions = evaluate(model,
                                                       dev_loader,
                                                       table_data,
                                                       args.beam_size)

        with open(os.path.join(output_path, 'predictions_sem_ql.json'), 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2)

        eval_results_string = "Epoch: {}    Sketch-Accuracy: {}     Accuracy: {}".format(epoch + 1, sketch_acc, acc)
        tqdm.write(eval_results_string)

        total_transformed, fail_transform, spider_eval_results = transform_to_sql_and_evaluate_with_spider(predictions,
                                                                                                           table_data,
                                                                                                           args.data_dir,
                                                                                                           output_path,
                                                                                                           tb_writer,
                                                                                                           epoch + 1)

        tqdm.write("Successfully transformed {} of {} from SemQL to SQL.".format(total_transformed - fail_transform, total_transformed))
        tqdm.write("Results from Spider-Evaluation:")
        for key, value in spider_eval_results.items():
            tqdm.write("{}: {}".format(key, value))

        if acc > best_acc:
            save_model(model, os.path.join(output_path))
            tqdm.write("Accuracy of this epoch ({}) is higher then the so far best accuracy ({}). Save model.".format(acc, best_acc))
            best_acc = acc

        with open(os.path.join(output_path, "eval_results.log"), "a+", encoding='utf-8') as writer:
            writer.write(eval_results_string + "\n")

        wandb.log({"Sketch-accuracy": sketch_acc, "accuracy": acc}, step=epoch + 1)

        tb_writer.add_scalar("sketch-accuracy", sketch_acc, epoch + 1)
        tb_writer.add_scalar("accuracy", acc, epoch + 1)

        scheduler.step()  # Update learning rate schedule

    tb_writer.close()
