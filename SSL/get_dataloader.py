import torch
import lightly.data as data

def get_dataloader(datafolder, transform, batch_size=224):
    """ Create standard dataloader

    Args:
        datafolder (str): path to data
        transform (lightly.transforms): type of transformations to apply to images
        batch_size (int, optional): batch size. Defaults to 224.

    Returns:
        _type_: dataloader
    """

    dataset = data.LightlyDataset(datafolder, transform=transform)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )

    return dataloader