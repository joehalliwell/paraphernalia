"""
Tools for working with `PyTorch Lightning <PL>`_.

.. _PL: https://www.pytorchlightning.ai/
"""

import io
import logging
import os
import warnings
from pathlib import Path

from torch.functional import Tensor

try:
    import ipywidgets as widgets  # type: ignore
except ImportError:
    warnings.warn("Could not import ipywidgets. Some functionality won't work")
    widgets = None

import imageio
import numpy as np
import pytorch_lightning as pl
import torch
from IPython.display import Image, display
from torchvision import transforms as T
from torchvision.utils import make_grid

_LOG = logging.getLogger(__name__)


class ImageCheckpoint(pl.Callback):
    """
    A PyTorch Lightning callback for saving and previewing images.

    Image batches (b, c, h, w) should be generated by
    `module.forward()`.
    """

    def __init__(
        self,
        path_template: str,
        video_path: str = None,
        interval=50,
        preview: bool = True,
    ):
        """
        `path_template` can draw on the following variables:

        - `index`: the index of the image in the provided batch
        - `model`: the Lightning model
        - `trainer`: the Lightning trainer

        Args:
            path_template (str): a path template as described above
            interval (int, optional): the checkpoint interval
            preview (bool, optional): if true display an ipywidget preview
                panel. Defaults to True.
        """
        super().__init__()
        self.path_template = str(path_template)
        self.interval = interval
        self._last_checkpoint_step = None
        self._preview = None
        self._video_writer = None

        _LOG.info(
            f"Checkpointing images to {self.path_template} every {self.interval} steps"
        )

        if video_path is not None:
            Path(video_path).parent.mkdir(parents=True, exist_ok=True)
            self._video_writer = imageio.get_writer(video_path)
            _LOG.info(f"Writing video to {video_path}")

        if preview:
            self._preview = widgets.Output()
            display(self._preview)

    def save(self, batch: Tensor, trainer: "pl.Trainer", module: "pl.LightningModule"):
        """Save the image batch."""
        if batch.shape[0] > 1 and "{index}" not in self.path.template:
            warnings.warn("Image batch size > 1, but template doesn't use {index}.")
            batch = make_grid(batch, nrow=4, padding=10)

        for i in range(batch.shape[0]):
            img = T.functional.to_pil_image(batch[i, :])
            filename = Path(
                str.format(self.path_template, module=module, trainer=trainer, index=i)
            )
            os.makedirs(filename.parent, exist_ok=True)
            img.save(filename)
            _LOG.info(f"Saved image as {filename}")

    def save_frame(self, batch: Tensor):
        if self._video_writer is None:
            return
        img = T.functional.to_pil_image(batch[0, :])
        img = np.array(img)
        self._video_writer.append_data(img)

    def preview(self, batch: Tensor):
        """Preview the image batch if configured, otherwise do nothing."""
        if not self._preview:
            return
        img = T.functional.to_pil_image(make_grid(batch, nrow=4, padding=10))
        # Workaround https://github.com/jupyter-widgets/ipywidgets/issues/3003
        b = io.BytesIO()
        img.save(b, format="PNG")
        img = Image(b.getvalue())

        # In principle could call clear_output. In practice the following works
        # better. See: <https://stackoverflow.com/a/64103274/3073930>
        self._preview.outputs = []
        self._preview.append_display_data(img)

    def checkpoint(self, trainer: "pl.Trainer", module: "pl.LightningModule"):
        """Main checkpoint function, called on epoch start and training end."""
        if trainer.global_step == self._last_checkpoint_step:
            return

        with torch.no_grad():
            module.eval()
            img = module.forward()
            module.train()

        self.save(img, trainer, module)
        self.save_frame(img)
        self.preview(img)
        self._last_checkpoint_step = trainer.global_step

    def on_batch_end(self, trainer: "pl.Trainer", module: "pl.LightningModule") -> None:
        """
        Called at the end of each batch.

        Checkpoints if a multiple of `self.interval`.
        """
        if (trainer.global_step % self.interval) == 0:
            self.checkpoint(trainer, module)

    def on_epoch_start(self, trainer: "pl.Trainer", module: "pl.LightningModule"):
        """
        Called at the start of an epoch.

        Always checkpoints.
        """
        self.checkpoint(trainer, module)

    def on_train_end(self, trainer: "pl.Trainer", module: "pl.LightningModule"):
        """
        Called when training ends.

        Always checkpoints.
        """
        _LOG.info("Shutting down ImageCheckpoint")
        self.checkpoint(trainer, module)
        if self._video_writer is not None:
            self._video_writer.close()
