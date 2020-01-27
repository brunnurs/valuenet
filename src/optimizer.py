from itertools import chain

from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR


def build_optimizer_encoder(model, num_train_steps, lr_transformer, lr_connection, lr_base, scheduler_gamma):
    print("Build optimizer and scheduler. Total training steps: {}".format(num_train_steps))

    # we use different learning rates for three set of parameters:
    # 1. fine-tuning the transformer.
    transformer_parameter = list(model.encoder.transformer_model.parameters())
    transformer_parameter_ids = list(map(lambda p: id(p), transformer_parameter))

    # 2. the parameters from layers connecting the transformer with the rest of the network
    connection_parameters = list(filter(lambda p: id(p) not in transformer_parameter_ids,
                                        chain(model.encoder.parameters(), model.decoder_cell_init.parameters())))

    connection_parameters_ids = list(map(lambda p: id(p), connection_parameters))

    # 3. all the remaining parameters
    remaining_parameters = list(filter(lambda p: id(p) not in transformer_parameter_ids + connection_parameters_ids,
                                       model.parameters()))

    assert len(transformer_parameter) + len(connection_parameters) + len(remaining_parameters) == len(list(model.parameters()))

    parameter_groups = [
        {'params': transformer_parameter, 'lr': lr_transformer},
        {'params': connection_parameters, 'lr': lr_connection},
        {'params': remaining_parameters, 'lr': lr_base},
    ]

    optimizer = Adam(parameter_groups)
    scheduler = MultiStepLR(optimizer, milestones=[21, 41, 61, 81], gamma=scheduler_gamma)

    return optimizer, scheduler
