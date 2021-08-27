import logging
from datetime import datetime

import pytorch_lightning as pl
import torch

import paraphernalia as pa
from paraphernalia.torch.lightning import ImageCheckpoint
from paraphernalia.utils import slugify

LOGGER = logging.getLogger(__name__)


class CriticSystem(pl.LightningModule):
    """A PyTorch Lightning Module for image generation."""

    def __init__(
        self, name: str, generator: torch.nn.Module, critic: torch.nn.Module, lr=0.5
    ):
        """
        Initialize the module.

        Args:
            name (str): name -- used to generate slugs
            generator (Module): an image generator
            critic (Module): a mechanism for rating generators
            lr (float, optional): [description]. Defaults to 0.5.
        """
        super().__init__()
        self.name = name
        self.generator = generator
        self.critic = critic
        self.lr = 0.5
        self.save_hyperparameters("name", "lr")

    def forward(self):
        return self.generator.forward()

    def training_step(self, batch, batch_idx):
        img = self.generator.forward()
        # sim = torch.utils.checkpoint.checkpoint(self.critic.forward, img)
        sim = self.critic(img)
        loss = 100 * (1.0 - sim.mean())
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """Sets up Adam optimizer."""
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        return optimizer

    def configure_callbacks(self):
        """Adds an image checkpoint."""
        dir = slugify(datetime.now().date(), self.name)
        size = f"{self.generator.size[0]}x{self.generator.size[1]}"
        filename = slugify(datetime.now(), self.name, size) + ".png"
        preview = ImageCheckpoint(
            pa.settings().project_home / dir / filename, preview=True
        )
        return [preview]

    def on_train_start(self):
        """Logs some info."""
        logging.info(f"Title: {self.name}")
        logging.info(f"Size: {self.generator.size}")
