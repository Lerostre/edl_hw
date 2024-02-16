import torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms
from torchvision.datasets import CIFAR10

from modeling.diffusion import DiffusionModel
from modeling.training import generate_samples, train_epoch, seed_everything
from modeling.unet import UnetModel

import hydra
from hydra.core.config_store import ConfigStore

# import wandbimport torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms
from torchvision.datasets import CIFAR10

from modeling.diffusion import DiffusionModel
from modeling.training import generate_samples, train_epoch, seed_everything
from modeling.unet import UnetModel

import hydra
from hydra.core.config_store import ConfigStore

import wandb
import config_def
import os
from omegaconf import OmegaConf

cs = ConfigStore.instance()
config_def.access_configs(cs)

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config: config_def.Config):
    seed_everything(config.train.random_seed)
    log_flag = config.db.log
    if log_flag:
        wandb.init(
            config=OmegaConf.to_container(config),
            project=config.db.project,
            name=str(config.db.name)
        )

    # init model
    if not torch.cuda.is_available():
        print("No cuda driver detected, switched to cpu")
        config.train.device = "cpu"
    device = config.train.device
    ddpm = DiffusionModel(
        eps_model=UnetModel(**config.model.unet),
        betas=config.model.betas,
        num_timesteps=config.model.num_timesteps,
    ).to(device)
    if log_flag:
        wandb.watch(ddpm)

    # augment data
    train_transforms = hydra.utils.instantiate(config.train.augmentation)
    dataset = CIFAR10(
        "cifar10",
        train=True,
        download=True,
        transform=train_transforms,
    )

    # init dataloaders
    subset_size = config.train.subset_size
    if isinstance(subset_size, float):
        subset_size = int(len(dataset) * subset_size)
    sampler = RandomSampler(dataset, num_samples=subset_size)
    dataloader = DataLoader(
        dataset, sampler=sampler,
        batch_size=config.train.loader.batch_size,
        num_workers=config.train.loader.num_workers,
    )
    
    # init optimizer with schedule
    optim = hydra.utils.instantiate(config.train.optimizer, params=ddpm.parameters())

    # train and generation loop
    if not os.path.exists("samples"):
        os.makedirs("samples")
    for i in range(config.train.num_epochs):
        train_epoch(ddpm, dataloader, optim, device, log=log_flag)
        generate_samples(ddpm, device, f"samples/{i:02d}.png", log=log_flag)


if __name__ == "__main__":
    main()
