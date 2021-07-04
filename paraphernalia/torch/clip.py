import logging
import random
from typing import List, Optional, Union

import clip
import torch
import torchvision.transforms as T
from torch.functional import Tensor

from paraphernalia.torch import overtile

logger = logging.getLogger(__name__)

TextOrTexts = Union[str, List[str]]


class CLIP(torch.nn.Module):

    _WINDOW_SIZE = 224
    _DETAIL_PROMPT_TEMPLATE = "Detail from a picture of {prompt}"

    def __init__(
        self,
        prompt: TextOrTexts,
        anti_prompt: Optional[TextOrTexts] = None,
        detail: Optional[TextOrTexts] = None,
        anti_detail: Optional[TextOrTexts] = None,
        use_tiling: bool = True,
        macro: float = 0.5,
        chops: int = 32,
        model: str = "ViT-B/32",
        device: Optional[str] = None,
    ):
        """
        A CLIP-based perceptor that evaluates how well an image fits with
        on or more target text prompts. Uses multiple scales to prevent
        aliasing effects, and allow high-resolution images to be processed.


        prompt:
            the text prompt to use in general

        anti_prompt:
            a description to avoid

        detail:
            a text prompt to use for micro-perception, defaults to "A detail from
            a picture of {prompt}"

        anti_detail:
            a description to avoid for micro-perception

        use_tiling: bool
            if true, add a covering of near-pixel-perfect perceptors into the
            mix

        chops: int
            augmentation operations
        """
        super(CLIP, self).__init__()

        # Value checks
        if model not in clip.available_models():
            raise ValueError(
                f"Invalid model. Must be one of: {clip.available_models()}"
            )

        if chops < 0:
            raise ValueError("Chops must be a strictly positive integer")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Prompt encoding
        self.device = torch.device(device)
        self.encoder, _ = clip.load(model, device=self.device)

        self._encoded_prompts = {}
        self.prompts = self._encode_texts(prompt, "prompts")
        if detail is None:
            detail = [
                self._DETAIL_PROMPT_TEMPLATE.format(prompt=prompt) for prompt in prompt
            ]
        self.detail_prompts = self._encode_texts(detail, "detail prompts")

        self.anti_prompts = self._encode_texts(anti_prompt, "anti-prompts")
        if anti_prompt and anti_detail is None:
            anti_detail = [
                self._DETAIL_PROMPT_TEMPLATE.format(prompt=prompt)
                for prompt in self.anti_prompts
            ]
        self.anti_details = self._encode_texts(anti_detail, "detail anti-prompts")

        # Image processing
        self.use_tiling = use_tiling
        self.chops = chops
        self.macro = macro

        # General input transformation, compare with the transform returned
        # as the second item by clip.load()
        self.transform = T.Compose(
            [
                # T.CenterCrop(size=self._WINDOW_SIZE),
                T.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                )
            ]
        )
        self.macro_transform = T.RandomResizedCrop(
            size=self._WINDOW_SIZE, scale=(0.95, 1.0), ratio=(1.0, 1.0)
        )
        self.micro_transform = T.RandomResizedCrop(
            size=self._WINDOW_SIZE, scale=(0.1, 0.5), ratio=(1.0, 1.0)
        )

    def _encode_texts(self, text_or_texts: str, what: str) -> dict[str, Tensor]:
        """
        Helper method used to initialize prompts.

        Args:
            text_or_texts (str): [description]
            what: a description of the group being encoded

        Returns:
            Tensor: A map from the (anti-)prompt texts to Tensors
        """
        if text_or_texts is None:
            logger.info(f"No {what}")
            return None

        elif isinstance(text_or_texts, str):
            texts_or_texts = [text_or_texts]

        encoded = self.encode_text(text_or_texts)
        logger.info(f"Encoded {len(text_or_texts)} {what}")
        return encoded

    def encode_text(self, text_or_texts: str) -> Tensor:
        """
        Encode text. Returns a detached tensor.
        """
        token_batch = clip.tokenize(text_or_texts).to(self.device)
        encoded = self.encoder.encode_text(token_batch)
        encoded = encoded.detach().clone()
        logger.debug(f"Encoded {len(text_or_texts)} texts")
        return encoded

    def encode_image(self, batch: Tensor) -> Tensor:
        return self.encoder.encode_image(batch)

    def lenses(self, img: Tensor) -> Tensor:
        macro = self.macro
        batch_size, c, h, w = img.shape

        if h < self._WINDOW_SIZE and w < self._WINDOW_SIZE:
            macro = 1.0

        macro_ops = int(self.macro * self.chops)
        micro_ops = int(self.chops - macro_ops)
        assert self.chops == (macro_ops + micro_ops)

        batch = []
        text_batch = []

        # Large random chops to manage composition and counteract aliasing
        for _ in range(macro_ops):
            batch.append(self.macro_transform(img))
            text_batch.append(self.prompts[0])

        # Small random pixel-perfect chops to focus on fine details
        for _ in range(micro_ops):
            batch.append(self.micro_transform(img))
            text_batch.append(self.detail_prompts[0])

        # Tiling of near-pixel-perfect chops
        if self.use_tiling:
            # tiling = tile(img, self._WINDOW_SIZE)
            tiling = overtile(img, int(self._WINDOW_SIZE * 1.1))
            tiling = self.macro_transform(tiling)
            num_tiles = tiling.shape[0] // batch_size
            batch.append(tiling)
            text_batch += [self.detail_prompts[0]] * num_tiles

        batch = torch.cat(batch, 0)

        text_batch = torch.cat(text_batch, 0)
        text_batch = text_batch.repeat(batch_size, 1)
        return batch, text_batch

    def forward(self, img: Tensor) -> Tensor:
        """
        Returns one loss for each image in the provided batch.

        TODO:
          - Enable micro/macro weighting beyond what we get naturally from chops
          - Add some kind of masking
        """

        batch_size = img.shape[0]

        img_batch, text_batch = self.lenses(img)
        img_batch = self.transform(img_batch)
        img_batch = self.encode_image(img_batch)

        losses = 1.0 - torch.cosine_similarity(img_batch, text_batch)

        # Split into a section per original batch
        per_image = torch.cat(
            [t.mean().unsqueeze(0) for t in torch.chunk(losses, batch_size)]
        )
        return per_image
