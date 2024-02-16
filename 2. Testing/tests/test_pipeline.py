import pytest
import torch
import numpy as np
import os
from torch.utils.data import DataLoader, RandomSampler
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import CIFAR10
from torch.optim import SGD, Adam

from modeling.diffusion import DiffusionModel
from modeling.training import train_step, train_epoch, generate_samples, seed_everything
from modeling.unet import UnetModel


@pytest.fixture
def train_dataset():
    transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=transforms,
    )
    return dataset


@pytest.mark.parametrize(["device"], [["cpu"], ["cuda"]])
def test_train_on_one_batch(device, train_dataset):
    if not torch.cuda.is_available():
        device = "cpu"
    # note: you should not need to increase the threshold or change the hyperparameters
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=32),
        betas=(1e-4, 0.02),
        num_timesteps=1000,
    )
    ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=5e-4)
    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    x, _ = next(iter(dataloader))
    loss = None
    for i in range(50):
        loss = train_step(ddpm, x, optim, device)
    assert loss < 0.5

# тут я нампаем сделал, потому что торч какой-то кривой в этом плане
# мне не нужно ничего генерить кудой, например, поэтому я возьму либу поудобнее
@pytest.mark.parametrize(
    ["device", "batch_size", "n_epochs", "subset_size",
     "lr", "hidden_size", "num_timesteps", "optim_name"],
    [(
        np.random.choice(["cpu", "cuda"]),
        np.random.randint(4, 128),
        np.random.randint(1, 10),
        np.random.choice([0.01, 0.02, 5, 10]),
        np.random.uniform(1e-5, 1e-1),
        np.random.choice([4, 8, 16, 32, 64, 128]),
        np.random.randint(1, 100),
        np.random.choice(["SGD", "Adam"])
    ) for _ in range(5)
    ]
)
def test_training(
    device, batch_size, n_epochs, subset_size,
    lr, hidden_size, num_timesteps, optim_name
):
    # note: implement and test a complete training procedure (including sampling)
    seed_everything(69)
    if not torch.cuda.is_available():
        device = "cpu"

    # init model
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=hidden_size),
        betas=(1e-4, 0.02),
        num_timesteps=num_timesteps
    ).to(device)

    # augment data
    train_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset = CIFAR10(
        "cifar10",
        train=True,
        download=True,
        transform=train_transforms,
    )

    # init dataloaders
    if isinstance(subset_size, float):
        subset_size = int(len(dataset) * subset_size)
    sampler = RandomSampler(dataset, num_samples=subset_size)
    dataloader = DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size,
        num_workers=4,
    )
    
    # init optimizer with schedule
    optim = Adam if optim_name == "Adam" else SGD
    optim = optim(ddpm.parameters(), lr=lr)

    # train and generation loop
    if not os.path.exists("samples"):
        os.makedirs("samples")
    for i in range(n_epochs):
        train_epoch(ddpm, dataloader, optim, device, log=False)
        generate_samples(ddpm, device, f"samples/{i:02d}.png", log=False)
