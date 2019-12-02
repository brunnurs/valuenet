from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup


def build_optimizer_encoder(model, num_train_steps, learning_rate, adam_eps, warmup_steps, weight_decay):
    print("Build optimizer and warmup scheduler. Total training steps: {}".format(num_train_steps))
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=num_train_steps)

    return optimizer, scheduler
