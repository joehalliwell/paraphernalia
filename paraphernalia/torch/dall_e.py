import dall_e
import PIL
import torch
import torchvision.transforms as T
from torchvision.utils import make_grid

from paraphernalia.utils import download


class DALL_E(torch.nn.Module):

    _NUM_CLASSES = 8192

    def __init__(self, tau=1.0, start=None, batch_size=1, latent=64, hard=False):
        """
        Image generator based on OpenAI's DALL-E release.

        batch_size: int
          How many independent latent vectors to use in parallel. This has a
          huge impact on memory use.
        """
        super(DALL_E, self).__init__()

        if batch_size < 1:
            raise ValueError("batch_size must be >0")

        self.tau = tau
        self.batch_size = batch_size
        self.latent = latent
        self.hard = hard

        self.decoder = dall_e.load_model(
            str(download("https://cdn.openai.com/dall-e/decoder.pkl")), "cuda"
        )

        # Initialization mode
        if start is None:
            # Nice terrazzo style noise
            z = torch.nn.functional.one_hot(
                torch.randint(0, self._NUM_CLASSES, (batch_size, latent, latent)),
                num_classes=self._NUM_CLASSES,
            )
            z = z.permute(0, 3, 1, 2)

        else:
            z = self.encode(start)
            z = torch.cat([z.detach().clone() for _ in range(batch_size)])

        # Force one-hot!
        z = torch.argmax(z, axis=1).cuda()
        z = (
            torch.nn.functional.one_hot(z, num_classes=self._NUM_CLASSES)
            .permute(0, 3, 1, 2)
            .float()
        )
        z = torch.log(z + 0.01 / self._NUM_CLASSES)
        self.z = torch.nn.Parameter(z)

    def forward(self):
        return self.generate()

    def generate(self, z=None, tau=None, hard=None):
        """
        Generate a batch of images.
        """
        if z is None:
            z = self.z
        if tau is None:
            tau = self.tau
        if hard is None:
            hard = self.hard

        samples = torch.nn.functional.gumbel_softmax(z, dim=1, tau=tau, hard=hard)

        buf = self.decoder(samples)
        buf = torch.sigmoid(buf[:, :3])
        buf = dall_e.unmap_pixels(buf.float())
        return buf

    def generate_image(self, **kwargs):
        """
        Convenience to generate a single PIL image within a no_grad block.
        """
        with torch.no_grad():
            imgs = self.generate(**kwargs)
            return T.functional.to_pil_image(make_grid(imgs, nrow=4, padding=10))

    def encode(self, img):
        """
        Encode an image or tensor.
        """
        if isinstance(img, PIL.Image.Image):
            img = PIL.ImageOps.pad(img, (self.latent * 8, self.latent * 8))
            img = torch.unsqueeze(T.functional.to_tensor(img), 0)

        with torch.no_grad():
            encoder = dall_e.load_model(
                str(download("https://cdn.openai.com/dall-e/encoder.pkl")), "cuda"
            )
            img = dall_e.map_pixels(img).cuda()
            z = encoder(img)

        return z.detach().clone()
