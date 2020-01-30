import torch
from tqdm import tqdm

from spider.example_builder import build_example


def train(global_step,
          tb_writer,
          train_dataloader,
          table_data,
          model,
          optimizer,
          clip_grad,
          sketch_loss_weight=1,
          lf_loss_weight=1):

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    model.train()

    for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
        examples = []
        for data_row in batch:
            try:
                example = build_example(data_row, table_data)
                examples.append(example)
            except RuntimeError as e:
                print("Exception while building example (training): {}".format(e))

        examples.sort(key=lambda e: -len(e.src_sent))

        sketch_loss, lf_loss = model.forward(examples)

        mean_sketch_loss = torch.mean(-sketch_loss)
        mean_lf_loss = torch.mean(-lf_loss)

        loss = lf_loss_weight * mean_lf_loss + sketch_loss_weight * mean_sketch_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        tr_loss += loss.item()

        optimizer.step()
        model.zero_grad()  # after we optimized the weights, we set the gradient back to zero.

        global_step += 1

        tb_writer.add_scalar('loss', (tr_loss - logging_loss), global_step)
        logging_loss = tr_loss

    return global_step
