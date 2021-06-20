from paraphernalia.utils import download
import einops
import torch
import PIL
import torchvision.transforms as T


class DALL_E(torch.nn.Module):
    def __init__(self, tau=1.0, start=None):
        """
        Image generator based on OpenAI's DALL-E release
        """
        import dall_e

        super(DALL_E, self).__init__()
        self.tau = tau
        self.decoder = dall_e.load_model(
            str(download("https://cdn.openai.com/dall-e/decoder.pkl")), "cuda"
        )
        self.encoder = dall_e.load_model(
            str(download("https://cdn.openai.com/dall-e/encoder.pkl")), "cuda"
        )

        if start is None:
            start = torch.randn(1, 3, 512, 512) * 0.5 + 0.5

        z = self.encode(start)
        # print(torch.min(z), torch.max(z), torch.mean(z), torch.std(z))
        self.z = torch.nn.Parameter(z.cuda())

        self.log_uniform = torch.log(torch.full((1, 64 * 64, 8192), 1.0 / 8192).cuda())

    def forward(self):
        return self.generate()

    def generate(self, z=None):
        if z is None:
            z = self.z
        z = torch.nn.functional.gumbel_softmax(z, dim=1, tau=self.tau)
        # z = torch.nn.functional.softmax(self.z, dim=1)

        buf = self.decoder(z)
        buf = torch.sigmoid(buf.float()[:, :3])
        buf = dall_e.unmap_pixels(buf)
        return buf

    def generate_image(self, z=None):
        if z is None:
            z = self.z
        with torch.no_grad():
            return T.ToPILImage(mode="RGB")(self.generate(z)[0])

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
