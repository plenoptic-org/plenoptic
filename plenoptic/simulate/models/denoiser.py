import torch
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import ssim, psnr

from adabelief_pytorch import AdaBelief


class Denoiser(pl.LightningModule):
    """

    Robust and interpretable blind image denoising via bias-free convolutional
    neural networks
    S Mohan*, Z Kadkhodaie*, E P Simoncelli and C Fernandez-Granda
    http://www.cns.nyu.edu/~lcv/pubs/makeAbs.php?loc=MohanKadkhodaie19b

    Solving linear inverse problems using the prior implicit in a denoiser
    Z Kadkhodaie and E P Simoncelli
    http://www.cns.nyu.edu/~lcv/pubs/makeAbs.php?loc=Kadkhodaie20a
    """
    def __init__(self, d=1, p=64, n_layers=12, relative_noise_std=0.8):
        super().__init__()

        # noise std, relative to signal std
        self.relative_noise_std = relative_noise_std

        self.convs = nn.ModuleList([nn.Conv2d(d, p, (3, 3),
                                    padding=1, bias=False)])
        # per channel target gain
        self.sigmas = nn.ModuleList([nn.Conv2d(p, p, (1, 1),
                                     groups=p, bias=False)])
        self.sigmas[-1].weight.data.fill_(1)
        # per channel source gain
        self.acc_std = [torch.ones(1, p, 1, 1)]
        for _ in range(1, n_layers-1):
            self.convs.append(nn.Conv2d(p, p, (3, 3), padding=1, bias=False))
            self.sigmas.append(nn.Conv2d(p, p, (1, 1), groups=p, bias=False))
            self.sigmas[-1].weight.data.fill_(1)
            self.acc_std.append(torch.ones(1, p, 1, 1))

        self.convs.append(nn.Conv2d(p, d, (3, 3), padding=1, bias=False))

        self.rho = nn.ReLU(inplace=True)

    def forward(self, y):
        for layer in range(len(self.convs)-1):
            y = self.convs[layer](y)
            if self.convs[layer].training:
                batch_std = torch.std(y, dim=(0, 2, 3), keepdims=True) + 1e-6
                y = y / batch_std
                self.acc_std[layer] = (1-.1) * self.acc_std[layer] + \
                                      (0.1) * batch_std.data
                y = self.sigmas[layer](y)
            else:
                y = y / self.acc_std[layer]
                y = self.sigmas[layer](y)
            y = self.rho(y)

        x_hat = self.convs[-1](y)
        return x_hat

    def corrupt(self, x):
        # per image noise strength
        noise_std = self.relative_noise_std * \
                    torch.std(x, dim=(1, 2, 3), keepdims=True)
        y = x + noise_std * torch.randn_like(x)
        return y

    def training_step(self, batch, batch_idx):
        x, c = batch
        y = self.corrupt(x)
        x_hat = self(y)

        loss = torch.nn.functional.mse_loss(x_hat, x)
        self.log('train_mse', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, c = batch
        y = self.corrupt(x)
        x_hat = self(y)

        loss = torch.nn.functional.mse_loss(x_hat, x)
        PSNR = psnr(x_hat, x)
        SSIM = ssim(x_hat, x)
        self.log('test_mse', loss, prog_bar=True)
        self.log('test_psnr', PSNR, prog_bar=True)
        self.log('test_ssim', SSIM, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        optimizer = AdaBelief(self.parameters(),
                              lr=1e-3, eps=1e-16, betas=(0.9, 0.999),
                              weight_decouple=True, rectify=False)

        return optimizer


# %%
# TODO move this to /training
# keep a good checkpoint there
if __name__ == "__main__":
    model = Denoiser(n_layers=16)
    # profiler="simple",
    # overfit_batches=.01,
    trainer = pl.Trainer(limit_train_batches=0.01,
                         limit_val_batches=0.01,
                         val_check_interval=.25,
                         max_epochs=2,
                         track_grad_norm=2,
                         terminate_on_nan=True)

    trainer.fit(model)
# %%
