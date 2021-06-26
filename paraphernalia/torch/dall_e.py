from paraphernalia.utils import download
import torch
import torchvision.transforms as T
import clip


class CLIP(torch.nn.Module):
    def __init__(
        self,
        text,
        detail_text=None,
        use_fovea=True,
        chops=32,
        macro=0.5,
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
        self.chops = chops
        self.macro = macro
        self.use_fovea = use_fovea

        # General input transformation
        self.window_size = 224  # Unlikely to change, but really linked to model
        self.transform = T.Compose(
            [
                T.CenterCrop(size=self.window_size),
                T.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        self.macro_transform = T.RandomResizedCrop(
            size=self.window_size, scale=(0.8, 1.0), ratio=(1.0, 1.0)
        )
        self.micro_transform = T.RandomCrop(size=self.window_size)

        self.encoder, _ = clip.load(model)
        with torch.no_grad():
            self.encoded_text = self.encode_text(text)
            self.encoded_detail_text = self.encode_text(detail_text)

    def encode_text(self, text):
        text = clip.tokenize(text).cuda()
        text = self.encoder.encode_text(text)
        text = text.detach().clone()
        return text

    def encode_image(self, batch):
        return self.encoder.encode_image(batch)

    def augment(self, img):
        img = self.cropper(img)
        return img

    def forward(self, img):
        """
        TODO:
          - Don't bother with this stuff if img.size < window
          - Enable micro/macro weighting beyond what we get natually from chops
          - Add foveal tiling
        """
        macro_ops = int(self.macro * self.chops)  # if img.size > window else 0
        micro_ops = int(self.chops - macro_ops)
        assert self.chops == (macro_ops + micro_ops)

        batch = []
        text_batch = []

        for _ in range(macro_ops):
            batch.append(self.macro_transform(img))
            text_batch.append(self.encoded_text)

        for _ in range(micro_ops):
            batch.append(self.micro_transform(img))
            text_batch.append(self.encoded_detail_text)

        batch = [self.transform(img) for img in batch]
        batch = torch.cat(batch, 0)
        batch = self.encode_image(batch)

        text_batch = torch.cat(text_batch, 0)

        loss = 1.0 - torch.cosine_similarity(text_batch, batch).mean()
        return loss
