import logging
from typing import List, Optional, Set, Tuple, Union

import clip
import torch
import torchvision.transforms as T
from torch.functional import Tensor

from paraphernalia.torch import cosine_similarity, overtile, regroup

logger = logging.getLogger(__name__)

TextOrTexts = Union[str, List[str]]


class CLIP(torch.nn.Module):
    """
    A CLIP-based perceptor that evaluates how well an image fits with
    on or more target text prompts.

    The underlying model is limited to (224, 224) resolution, so this
    class presents it with multiple perspectives on an image:

    * Macro: random crops of 90-100% of the image, used to counteract aliasing
    * Micro: small near-pixel-perfect random crops, and an optional tiling to enable
      the fine details of high-resolution images to be processed.

    A lot of internals are exposed via methods to facilitate debugging and
    experimentation.

    Args:
        prompt:
            the text prompt to use in general

        anti_prompt:
            a description to avoid

        detail:
            a text prompt to use for micro-perception, defaults to "A detail from
            a picture of {prompt}"

        anti_detail:
            a description to avoid for micro-perception

    Attributes:
        use_tiling (bool):
            if true, add a covering of near-pixel-perfect perceptors into the
            mix

        chops (int):
            augmentation operations, these get split 50-50 between macro and micro
    """

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
        chops: int = 64,
        model: str = "ViT-B/32",
        device: Optional[str] = None,
    ):
        super(CLIP, self).__init__()

        # Value checks
        if model not in clip.available_models():
            raise ValueError(
                f"Invalid model. Must be one of: {clip.available_models()}"
            )

        if macro < 0.0 or macro > 1.0:
            raise ValueError("Macro must be between 0.0 and 1.0")

        if chops < 0:
            raise ValueError("Chops must be a strictly positive integer")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Prompt encoding
        self.device = torch.device(device)
        # self.encoder, _ = clip.load(model, device=self.device)
        self.encoder = (
            clip.load(model, jit=False)[0].eval().requires_grad_(False).to(device)
        )

        self._encoded_prompts = {}
        self.prompts, prompts = self._encode_texts(prompt, "prompts")
        if detail is None:
            detail = [
                self._DETAIL_PROMPT_TEMPLATE.format(prompt=prompt) for prompt in prompts
            ]
        self.detail_prompts, _ = self._encode_texts(detail, "detail prompts")

        self.anti_prompts, _ = self._encode_texts(anti_prompt, "anti-prompts")
        if anti_prompt and anti_detail is None:
            anti_detail = [
                self._DETAIL_PROMPT_TEMPLATE.format(prompt=prompt)
                for prompt in self.anti_prompts
            ]
        self.anti_details, _ = self._encode_texts(anti_detail, "detail anti-prompts")

        # Macro vs micro weighting
        self.macro = macro

        # Image processing
        self.use_tiling = use_tiling
        self.chops = chops

        # General input transformation, compare with the transform returned
        # as the second item by clip.load()
        self.transform = T.Compose(
            [
                # Ensure that images are window-sized. Not really necessary, but
                # harmless and means that encode_image is more flexible.
                T.CenterCrop(size=self._WINDOW_SIZE),
                T.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        self.macro_transform = T.Compose(
            [
                # T.ColorJitter(saturation=0.01, brightness=0.01, hue=0.01),
                # T.RandomPerspective(distortion_scale=0.05, p=0.5),
                T.RandomResizedCrop(
                    size=self._WINDOW_SIZE, scale=(0.9, 1.0), ratio=(1.0, 1.0)
                ),
                T.RandomHorizontalFlip(p=0.5),
            ]
        )
        self.micro_transform = T.Compose(
            [
                # T.ColorJitter(saturation=0.1, brightness=0.1, hue=0.1),
                # T.RandomPerspective(distortion_scale=0.2, p=0.5),
                T.RandomCrop(size=self._WINDOW_SIZE),
                T.RandomHorizontalFlip(p=0.5),
            ]
        )
        # self.micro_transform = T.RandomResizedCrop(
        #     size=self._WINDOW_SIZE, scale=(0.2, 0.5), ratio=(1.0, 1.0)

    def _encode_texts(self, text_or_texts: str, what: str) -> Tuple[Tensor, Set[str]]:
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
            return None, None

        elif isinstance(text_or_texts, str):
            text_or_texts = [text_or_texts]

        text_or_texts = set(text_or_texts)
        encoded = self.encode_text(text_or_texts)
        logger.info(f"Encoded {len(text_or_texts)} {what}")
        return encoded, text_or_texts

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
        batch = self.transform(batch)
        return self.encoder.encode_image(batch)

    def get_macro(self, img: Tensor) -> Tensor:
        n = self.chops - self.chops // 2
        return regroup([self.macro_transform(img) for _ in range(n)])

    def get_micro(self, img: Tensor) -> Tensor:

        # Small random pixel-perfect chops to focus on fine details
        n = self.chops // 2
        micro_batch = [self.micro_transform(img) for _ in range(n)]

        # (Optionally) Tiling of near-pixel-perfect chops
        if self.use_tiling:
            tiling = overtile(img, int(self._WINDOW_SIZE * 1.1), 0.5)
            micro_batch.extend(self.micro_transform(tile) for tile in tiling)

        return regroup(micro_batch)

    def get_similarity(
        self, img: Tensor, prompts: Tensor, batch_size: int, match="all"
    ) -> Tensor:
        """
        Compute the average similarity between a combined but contiguous
        batch of images and set of prompts.

        Args:
            imgs (Tensor): A combined-but-contiguous image batch with shape (batch_size * t, c, h, w)
            prompts (Tensor): A tensor of prompt embeddings with shape (n, 512)
            batch_size (int): The size of the original image batch
            match (str): Policy for multiple prompts. "any", "all" or (in future) "one"

        Returns:
            Tensor: A tensor of average similarities with shape (batch_size,)
        """
        assert img.shape[0] % batch_size == 0  # Must be a multiple
        encoded = self.encode_image(img)
        similarity = cosine_similarity(encoded, prompts)

        # Aggregate over prompts
        if match == "all":
            # Match all prompts equally
            # Should be similarity = torch.mean(dim=1)
            # But that's can be rolled into the chunk mean below
            pass
        elif match == "any":
            # Pick the best match -- intended for details
            similarity = torch.max(similarity, dim=1)[0]
        else:
            raise ValueError(f"Unknown matching mode {match}")

        # Average over perceptual windows to produce a similarity per image in the batch
        means = [chunk.mean() for chunk in torch.chunk(similarity, chunks=batch_size)]
        assert len(means) == batch_size
        return torch.stack(means)

    def forward(self, img: Tensor) -> Tensor:
        """
        Returns a similarity (0, 1) for each image in the provided batch.

        TODO:
          - Enable micro/macro weighting beyond what we get naturally from chops
          - Add some kind of masking

        Args:
            img (Tensor): A (b, c, h, w) image tensor

        Returns:
            Tensor: A vector of size b
        """
        batch_size, _c, h, w = img.shape

        macro_batch = self.get_macro(img)
        prompt_similarity = self.get_similarity(
            macro_batch, self.prompts, batch_size=batch_size
        )

        # If the window covers most of the image, use macro prompts
        ratio = min(self._WINDOW_SIZE / h, self._WINDOW_SIZE / w)
        detail_prompts = self.detail_prompts if ratio < 0.5 else self.prompts

        micro_batch = self.get_micro(img)
        detail_similarity = self.get_similarity(
            micro_batch, detail_prompts, batch_size=batch_size, match="any"
        )

        return self.macro * prompt_similarity + (1 - self.macro) * detail_similarity
