from tqdm import tqdm

from src.input_features import tokenize_input


def train(device,
          train_dataloader,
          model,
          tokenizer,
          optimizer,
          scheduler,
          max_seq_length):
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        model.train()

        input_ids, attention_mask, segment_ids = tokenize_input(batch, tokenizer, max_seq_length, device)