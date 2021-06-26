import clip
import torch
import torchvision.transforms as T

from paraphernalia.torch import tile
from paraphernalia.utils import download


class CLIP(torch.nn.Module):

    _WINDOW_SIZE = 224

    def __init__(
        self,
        text,
        detail_text=None,
        use_tiling=False,
        macro=0.5,
        chops=16,
        model="ViT-B/32",
    ):
        """
        A CLIP-based perceptor that evaluates how well an image fits with
        on or more target text prompts. Perception is batched for efficiency.

        text: str
            the text prompt to use in general

        detail_text: str
            a text prompt to use for micro perception, defaults to "A fragment
            of a picture of {text}"

        use_tiling: bool
            if true, add an optimla tiling of pixel-perfect perceptors into the
            mix

        chops: int
            augmentation operations
        """
        super(CLIP, self).__init__()
        if detail_text is None:
            detail_text = f"A fragment of a picture of {text}"

        if model not in clip.available_models():
            raise ValueError(
                f"Invalid model. Must be one of: {clip.available_models()}"
            )

        if chops < 0:
            raise ValueError("Chops must be a strictly positive integer")

        self.text = text
        self.detail_text = detail_text
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
                ),
            ]
        )
        self.macro_transform = T.RandomResizedCrop(
            size=self._WINDOW_SIZE, scale=(0.95, 1.0), ratio=(1.0, 1.0)
        )
        self.micro_transform = T.RandomCrop(size=self._WINDOW_SIZE)

        # Encode text
        self.encoder, _ = clip.load(model)
        self.encoded_text = self.encode_text(text)
        self.encoded_detail_text = self.encode_text(detail_text)

    def encode_text(self, text: str):
        """
        Encode text. Returns a detached tensor.
        """
        text = clip.tokenize(text).cuda()
        text = self.encoder.encode_text(text)
        text = text.detach().clone()
        return text

    def encode_image(self, batch):
        return self.encoder.encode_image(batch)

    def forward(self, img, mask=None):
        """
        TODO:
          - Don't bother with this stuff if img.size < window
          - Enable micro/macro weighting beyond what we get natually from chops
        """

        b, c, h, w = img.shape

        # Special case for starter batches
        if h == self._WINDOW_SIZE and w == self._WINDOW_SIZE:
            return 1.0 - torch.cosine_similarity(
                self.encoded_text, self.encode_image(img)
            )

        batch = []
        text_batch = []

        macro_ops = int(self.macro * self.chops)  # if img.size > window else 0
        micro_ops = int(self.chops - macro_ops)
        assert self.chops == (macro_ops + micro_ops)

        # Large random chops to manage composition and counteract aliasing
        for _ in range(macro_ops):
            batch.append(self.macro_transform(img))
            text_batch.append(self.encoded_text)

        # Small random pixel-perfect chops to focus on fine details
        for _ in range(micro_ops):
            batch.append(self.micro_transform(img))
            text_batch.append(self.encoded_detail_text)

        # Tiling of pixel-perfect chops
        if self.use_tiling:
            tiling = tile(img, self._WINDOW_SIZE)
            batch.append(tiling)
            text_batch += [self.encoded_detail_text] * tiling.shape[0]

        batch = [self.transform(img) for img in batch]
        batch = torch.cat(batch, 0)
        batch = self.encode_image(batch)

        text_batch = torch.cat(text_batch, 0)

        loss = 1.0 - torch.cosine_similarity(text_batch, batch)
        return loss
