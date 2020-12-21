import torch
import wandb
from tqdm import tqdm

from spider.example_builder import build_example


def train(global_step,
          train_dataloader,
          schema,
          model,
          optimizer,
          clip_grad,
          sketch_loss_weight=1,
          lf_loss_weight=1):

    tr_loss = 0.0
    model.zero_grad()
    model.train()

    for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
        examples = []
        for data_row in batch:
            try:
                example = build_example(data_row, schema)
                examples.append(example)
            except RuntimeError as e:
                print("Exception while building example (training): {}".format(e))

        examples.sort(key=lambda e: -len(e.question_tokens))

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

    return global_step
