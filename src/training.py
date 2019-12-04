import torch
from tqdm import tqdm

from src.input_features import tokenize_input


def train(global_step,
          tb_writer,
          device,
          train_dataloader,
          encoder_model,
          tokenizer,
          optimizer,
          scheduler,
          label_map,
          max_seq_length):
    tr_loss, logging_loss = 0.0, 0.0
    encoder_model.zero_grad()

    for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
        encoder_model.train()

        input_ids, attention_mask, segment_ids, label_ids = tokenize_input(batch,
                                                                           label_map,
                                                                           tokenizer,
                                                                           max_seq_length,
                                                                           device)

        outputs = encoder_model(input_ids, attention_mask=attention_mask, token_type_ids=segment_ids, labels=label_ids)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder_model.parameters(), 1.0)

        tr_loss += loss.item()

        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        encoder_model.zero_grad()  # after we optimized the weights, we set the gradient back to zero.

        global_step += 1

        tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
        tb_writer.add_scalar('loss', (tr_loss - logging_loss), global_step)
        logging_loss = tr_loss

    return global_step
