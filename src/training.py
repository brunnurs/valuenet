import numpy as np
import torch
from tqdm import tqdm

from src.spider.example_builder import build_example


def train(global_step,
          tb_writer,
          sql_data,
          table_data,
          model,
          optimizer,
          clip_grad,
          sketch_loss_weight=1,
          lf_loss_weight=1):

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    perm=np.random.permutation(len(sql_data))
    cum_loss = 0.0
    st = 0

    # for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
    while st < len(sql_data):
        model.train()

        ed = st + 64 if st + 64 < len(perm) else len(perm)

        examples = []
        for idx in range(st, ed):
            try:
                example = build_example(sql_data[perm[idx]], table_data)
                examples.append(example)
            except RuntimeError as e:
                print(str(e))

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

        cum_loss += loss.data.cpu().numpy()*(ed - st)
        st = ed

    return global_step
