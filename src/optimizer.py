from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR


def build_optimizer_encoder(model, num_train_steps, learning_rate, scheduler_gamma):
    print("Build optimizer and warmup scheduler. Total training steps: {}".format(num_train_steps))
    optimizer = Adam(model.parameters(), lr=learning_rate)
    # This milestones look a bit random to me, have taken it over from the IRNet paper. Maybe we need to adapt the schedule
    # once we use transformers again.
    scheduler = MultiStepLR(optimizer, milestones=[21, 41], gamma=scheduler_gamma)

    return optimizer, scheduler
