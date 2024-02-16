import torch
import inspect
from torch.optim.optimizer import Optimizer
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm

import wandb
from modeling.diffusion import DiffusionModel

# это чтобы по максимуму сид фиксировать, вообще зря оно здесь, coverage ломает
def seed_everything(seed):
    if seed is not None:
        seed = int(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

# это костыль, чтобы юзеру было удобнее какой-то оптимайзер задавать
# def get_optimizer(
#     optim_name: str = "adam",
#     model_params=None,
#     optimizer_params: dict = {},
# ):
#     if optim_name.lower() == "adam":
#         optimizer = Adam
#     elif optim_name.lower() == "sgd":
#         optimizer = SGD
#     else:
#         raise Exception("Only Adam ang SGD optimizers are supported")
#     optimizer_params = {
#         param: optimizer_params[param]
#         for param in inspect.signature(optimizer).parameters
#         if param in optimizer_params
#     }
#     return optimizer(model_params, **optimizer_params)

def train_step(model: DiffusionModel, inputs: torch.Tensor, optimizer: Optimizer, device: str):
    optimizer.zero_grad()
    inputs = inputs.to(device)
    loss = model(inputs)
    loss.backward()
    optimizer.step()
    return loss


def train_epoch(
    model: DiffusionModel,
    dataloader: DataLoader, 
    optimizer: Optimizer, 
    device: str,
    log: bool = False
):
    model.train()
    pbar = tqdm(dataloader)
    loss_ema = None
    is_first_batch = True
    for x, _ in pbar:
        train_loss = train_step(model, x, optimizer, device)
        # che takoe loss_ema
        loss_ema = train_loss if loss_ema is None else 0.9 * loss_ema + 0.1 * train_loss
        pbar.set_description(f"loss: {loss_ema:.4f}")
        if log:
            storage = dict()
            if is_first_batch:
                storage["inputs"] = x
                is_first_batch = False
            storage["train_loss"] = loss_ema
            wandb.log(storage)
            # мб можно lr чекать, но у меня нет шедулера, зачем?
            # можно градиенты отсматривать, но как-то лень, вроде стабильно всё


def generate_samples(
    model: DiffusionModel,
    device: str, 
    path: str,
    log: bool = False
):
    model.eval()
    with torch.no_grad():
        noise, samples = model.sample(8, (3, 32, 32), device=device)
        sample_grid = make_grid(samples, nrow=4)
        noise_grid = make_grid(noise, nrow=4)
        save_image(sample_grid, path)
        if log:
            image_name = path.split("/")[1]
            noise = wandb.Image(noise_grid, caption=f"noise_{image_name}")
            image = wandb.Image(sample_grid, caption=image_name)
            wandb.log({"noise": noise, "sampled": image})