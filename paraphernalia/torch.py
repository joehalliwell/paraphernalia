from paraphernalia.utils import download
import einops
import torch
import PIL
import torchvision.transforms as T


class DALL_E(torch.nn.Module):
    def __init__(self, tau=0.1, start=None, batch_size=1):
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

        if start is None:
            z = torch.randn(1, 8192, 64, 64)
        else:
            z = self.encode(start)

        self.z = torch.nn.Parameter(z.cuda())

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

        # TODO: Vary tau across batch?
        z = torch.nn.functional.log_softmax(z, dim=1)
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
            img = PIL.ImageOps.pad(img, (512, 512))
            img = torch.unsqueeze(T.ToTensor()(img), 0)

        with torch.no_grad():
            img = dall_e.map_pixels(img).cuda()
            z = self.encoder(img)

        return z.detach().clone()

    def compute_loss(self):
        """
        Experimental regularization loss: stay close to uniform
        Not used.
        """
        log_z = torch.nn.functional.log_softmax(self.z, dim=1)
        log_z = einops.rearrange(log_z, "b n hw -> b hw n")
        loss = torch.nn.functional.kl_div(
            log_z, self.log_uniform, reduction="batchmean", log_target=True
        )
        return loss
