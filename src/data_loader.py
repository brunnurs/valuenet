import torch


def get_data_loader(data_train, data_dev, batch_size, shuffle_train=True, shuffle_dev=False):
    train_loader = torch.utils.data.DataLoader(
        batch_size=batch_size,
        dataset=data_train,
        shuffle=shuffle_train,
        collate_fn=lambda x: x  # now dictionary values are not merged!
    )

    dev_loader = torch.utils.data.DataLoader(
        batch_size=batch_size,
        dataset=data_dev,
        shuffle=shuffle_dev,
        collate_fn=lambda x: x  # now dictionary values are not merged!
    )

    return train_loader, dev_loader
