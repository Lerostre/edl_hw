import torch
import torch.nn as nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, residual: bool = False):
        super().__init__()
        # паддинг не нужен, на одну свёртку больше
        # или нужен?
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
        )

        self.is_res = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.main(x)
        if self.is_res:
            x = x + self.conv(x)
            return x / 1.414
        else:
            return self.conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # не было страйда
        # в другом порядке
        self.layers = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ConvBlock(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # тут как-то логика совсем нарушена
        # размерности тензоров не совпадут, склеить не выйдет
        # сначала апсэмпл, потом допаддить, потом склеить
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        # здесь тогда тоже другая размерность
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # вот тут я подсмотрел - https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
        # ну иначе тут просто форвард неправильный
        x = self.upsample(x)

        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x,
            [diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2]
        )

        x = torch.cat((x, skip), 1)
        x = self.conv(x)

        return x


class TimestepEmbedding(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()

        self.lin1 = nn.Linear(1, emb_dim, bias=False)
        self.lin2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1)
        x = torch.sin(self.lin1(x))
        x = self.lin2(x)
        # unsqueeze нужен
        return x.view(-1, x.shape[1], 1, 1)


class UnetModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_size: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.hidden_size = hidden_size

        # residual
        self.init_conv = ConvBlock(in_channels, hidden_size, residual=False)

        # 4e s kanalami?
        self.down1 = DownBlock(hidden_size, 2*hidden_size)
        self.down2 = DownBlock(2*hidden_size, 4*hidden_size)
        self.down3 = DownBlock(4*hidden_size, 8*hidden_size)

        # po4 avg pool i 4?
        # self.to_vec = nn.Sequential(nn.AvgPool2d(4), nn.ReLU())
        # self.to_vec = nn.Sequential(nn.MaxPool2d(2), nn.ReLU())
        self.to_vec = DownBlock(8*hidden_size, 16*hidden_size)

        # здесь тоже другой размер?
        self.timestep_embedding = TimestepEmbedding(16 * hidden_size)

        # если я правильно понял, их надо на каждый слой наклепать
        self.te_init = TimestepEmbedding(hidden_size)
        self.te_down1 = TimestepEmbedding(2*hidden_size)
        self.te_down2 = TimestepEmbedding(4*hidden_size)
        self.te_down3 = TimestepEmbedding(8*hidden_size)
        self.te_up1 = TimestepEmbedding(16*hidden_size)
        self.te_up2 = TimestepEmbedding(8*hidden_size)
        self.te_up3 = TimestepEmbedding(4*hidden_size)
        self.te_up4 = TimestepEmbedding(2*hidden_size)

        # почему 4, 4?
        # self.up0 = nn.Sequential(
        #     nn.ConvTranspose2d(16*hidden_size, 8*hidden_size, 2, 2),
        #     nn.GroupNorm(8, 8*hidden_size),
        #     nn.ReLU(),
        # )

        # ап блоки вообще неправильные - не те размерности, плюс первый апблок тоже нормальный
        # ещё один ап нужен
        self.up1 = UpBlock(16*hidden_size, 8*hidden_size)
        self.up2 = UpBlock(8*hidden_size, 4*hidden_size)
        self.up3 = UpBlock(4*hidden_size, 2*hidden_size)
        self.up4 = UpBlock(2*hidden_size, hidden_size)
        self.out = nn.Conv2d(hidden_size, self.out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # print("original", x.shape)
        x = self.init_conv(x) + self.te_init(t)
        # print("init_conv", x.shape)

        down1 = self.down1(x) + self.te_down1(t)
        # print("down1", down1.shape)
        down2 = self.down2(down1) + self.te_down2(t)
        # print("down2", down2.shape)
        down3 = self.down3(down2) + self.te_down3(t)
        # print("down3", down3.shape)
        thro = self.to_vec(down3) + self.te_up1(t)
        # print("thro", thro.shape)
        # нужен бродкаст
        # temb = self.timestep_embedding(t).unsqueeze(-1).unsqueeze(-1)
        # print("temb", temb.shape)
        # thro = self.up0(thro + temb)
        # print("thro", thro.shape)
        # print(self.up1(thro, down3).shape)
        up1 = self.up1(thro, down3) + self.te_up2(t)
        # print("up1", up1.shape)
        up2 = self.up2(up1, down2) + self.te_up3(t)
        # print("up2", up2.shape, down1.shape)
        up3 = self.up3(up2, down1) + self.te_up4(t)
        # print("up3", up3.shape)
        up4 = self.up4(up3, x)
        # print("up3", up4.shape)

        # зачем конкатенация в конце?
        # out = self.out(torch.cat((up3, x), 1))
        out = self.out(up4)
        # print("out", out.shape)

        return out
