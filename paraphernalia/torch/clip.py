from paraphernalia.utils import download
import einops
import torch
import PIL
import torchvision.transforms as T


class CLIP(torch.nn.Module):
    def __init__(self, text, chops=32, model="ViT-B/32"):
        """
        chops: augmentation operations
        """
        super(CLIP, self).__init__()
        self.encoder, _ = clip.load(model)
        self.transform = T.Compose(
            [
                # T.Resize(size=224, interpolation=PIL.Image.BICUBIC),
                T.CenterCrop(size=224),
                T.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        self.cropper = T.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(1.0, 1.0))
        self.text = text
        self.encoded_text = self.encode_text(text)
        self.chops = chops
        self.macro = 1.0

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
        batch = []
        macro_ops = int(self.macro * self.chops)
        batch += [self.augment(img) for _ in range(macro_ops)]
        batch += [
            T.RandomCrop(
                224,
            )(img)
            for _ in range(self.chops - macro_ops)
        ]

        batch = [self.transform(img) for img in batch]
        batch = torch.cat(batch, 0)
        batch = self.encode_image(batch)

        loss = 1.0 - torch.cosine_similarity(self.encoded_text, batch)
        return loss
