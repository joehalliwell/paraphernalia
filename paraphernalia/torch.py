from paraphernalia.utils import download
import einops
import torch
import PIL
import torchvision.transforms as T


class DALL_E(torch.nn.Module):
    def __init__(self, tau=0.1, start=None, batch_size=1, latent=64):
        """
        Image generator based on OpenAI's DALL-E release.
        """
        super(DALL_E, self).__init__()
        self.decoder = dall_e.load_model(
            str(download("https://cdn.openai.com/dall-e/decoder.pkl")), "cuda"
        )
        self.encoder = dall_e.load_model(
            str(download("https://cdn.openai.com/dall-e/encoder.pkl")), "cuda"
        )

        self.tau = tau
        self.batch_size = batch_size
        self.latent = latent

        # Initialization mode
        if start is None:
            z = torch.nn.functional.one_hot(
                torch.full((latent, latent), 666), num_classes=8192
            )
            z = z.permute(2, 0, 1).unsqueeze(0).float().cuda()
        else:
            z = self.encode(start)

        # Ensure scaling is consistent
        z = torch.nn.functional.log_softmax(z, dim=1)
        z += torch.randn(1, 8192, self.latent, self.latent).cuda()

        self.z = torch.nn.Parameter(z)

    def forward(self):
        return self.generate()

    def generate(self, z=None, tau=None, batch_size=None):
        """
        Generate a batch of images.
        """
        if z is None:
            z = self.z
        if tau is None:
            tau = self.tau
        if batch_size is None:
            batch_size = self.batch_size

        # Create batch_size samples
        samples = torch.cat(
            [
                torch.nn.functional.gumbel_softmax(z, dim=1, tau=tau)
                for _ in range(batch_size)
            ]
        )
        samples = samples.float()

        buf = self.decoder(samples)
        buf = buf.float()
        buf = torch.sigmoid(buf.float()[:, :3])
        buf = dall_e.unmap_pixels(buf)
        return buf

    def generate_image(self, **kwargs):
        """
        Convenience to generate a single PIL image.
        """
        with torch.no_grad():
            img = self.generate(batch_size=1, **kwargs)[0]
            return T.ToPILImage(mode="RGB")(img)

    def encode(self, img):
        """
        Encode an image or tensor.
        """
        if isinstance(img, PIL.Image.Image):
            img = PIL.ImageOps.pad(img, (self.latent * 8, self.latent * 8))
            img = torch.unsqueeze(T.ToTensor()(img), 0)

        with torch.no_grad():
            img = dall_e.map_pixels(img).cuda()
            z = self.encoder(img)

        return z.detach().clone()
