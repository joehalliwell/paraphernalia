"""
Render GLSL fragment shaders to screen/video.
"""
import logging
import sys
from pathlib import Path
from time import time

import click
import imageio
import moderngl
import numpy as np
from moderngl_window import create_window_from_settings, settings
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
_LOG = logging.getLogger(__name__)
_LOG.setLevel(logging.INFO)


class FakeUniform:
    """
    Used to provide a uniform, where a shader doesn't have one. Has no effect.
    """

    value = None


"""
Vertex shader used for all rendering.
"""
VERTEX_SHADER = """
    #version 330
    in vec2 vx;
    out vec2 uv;

    void main() {
        gl_Position = vec4(vx, 0.0, 1.0);
        uv = vx;
    }
    """


class Renderer:
    """
    Core logic for rendering fragment shaders, shared across
    render and preview functions.
    """

    def __init__(
        self, ctx, fragment_shader: str, resolution=(100, 100), duration=0
    ) -> None:
        """
        Initializer.

        Args:
            ctx ([type]): Context provided by ModernGL
            fragment_shader (str): A fragment shader
            resolution (tuple, optional): [description]. Defaults to (100, 100).
            duration (int, optional): [description]. Defaults to 0.
        """
        self.ctx = ctx
        self.prog = ctx.program(
            vertex_shader=VERTEX_SHADER, fragment_shader=fragment_shader
        )

        vertices = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0])
        self.vbo = self.ctx.buffer(vertices.astype("f4"))
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, "vx")

        # Uniforms
        self.u_time = self.prog.get("u_time", FakeUniform())
        self.u_time.value = 0

        self.u_resolution = self.prog.get("u_resolution", FakeUniform())
        self.u_resolution.value = resolution

        self.u_duration = self.prog.get("u_duration", FakeUniform())
        self.u_duration.value = duration

        self.u_mouse = self.prog.get("u_mouse", FakeUniform())
        self.u_mouse.value = (0, 0)

        # TODO: buffers, absolute time

    def render(self, time):
        """Draw to the currently bound framebuffer

        Args:
            time (float): Elapsed time in seconds
        """
        self.u_time.value = time
        self.ctx.clear(0.0, 1.0, 0.0)
        self.vao.render(moderngl.TRIANGLE_STRIP)


@click.command()
@click.argument("fragment_shader", type=click.Path(exists=True, readable=True))
@click.option("--width", type=int, default=1024, help="Target width", show_default=True)
@click.option(
    "--height", type=int, default=1024, help="Target height", show_default=True
)
@click.option(
    "--duration",
    type=float,
    default=30,
    help="Target duration, loop if > 0",
    show_default=True,
)
@click.option("--speed", type=float, default=1.0, help="Multiplier for elapsed time")
@click.option(
    "--watch/--no-watch",
    default=True,
    help="Automatically reload when shader changes on disk",
)
@click.option("--scale", type=float, default=1.0, help="TODO", show_default=True)
def preview(
    fragment_shader, width, height, duration, speed, watch, scale
):  # pragma: no cover
    """
    Show a fragment shader in a window.
    """
    loop = duration > 0

    fragment_shader = Path(fragment_shader)
    settings.WINDOW.update(
        {
            "gl_version": (3, 3),
            "title": fragment_shader.name,
            "size": (width, height),
            "aspect_ratio": 1.0,
            "resizable": False,
        }
    )

    window = create_window_from_settings()
    last_mtime = -1
    mtime = 0
    renderer = None
    start = time()

    while not window.is_closing:
        # Watch the file and reload as needed
        if watch:
            mtime = fragment_shader.stat().st_mtime
        if mtime > last_mtime:
            logging.info(f"Loading {fragment_shader}")
            try:
                renderer = Renderer(
                    window.ctx,
                    fragment_shader=fragment_shader.open("r").read(),
                    resolution=window.size,
                    duration=duration,
                )
            except Exception:
                logging.exception(f"Failed to load {fragment_shader}")
                if renderer is None:
                    logging.critical("Aborting")
                    sys.exit(1)
            last_mtime = mtime
            # If looping, then
            if loop:
                start = time()

        renderer.u_mouse.value = window._mouse_pos

        window.use()
        elapsed = (time() - start) * speed
        renderer.render(elapsed % duration if loop else elapsed)
        window.swap_buffers()


@click.command()
@click.argument("fragment_shader", type=click.File("r"))
@click.option(
    "-o",
    "--output",
    type=click.Path(writable=True),
    default="output.mp4",
    help="Output file",
    show_default=True,
)
@click.option(
    "--width", type=int, default=1024, help="Width in pixels", show_default=True
)
@click.option(
    "--height", type=int, default=1024, help="Height in pixels", show_default=True
)
@click.option(
    "--fps", type=float, default=25.0, help="Frames per second", show_default=True
)
@click.option(
    "--duration",
    type=float,
    default=30.0,
    help="Duration in seconds",
    show_default=True,
)
@click.option(
    "--quality", type=int, default=5, help="Encoder quality", show_default=True
)
def render(fragment_shader, output, width, height, fps, duration, quality):
    """
    Render a fragment shader to an MP4 file.
    """

    # TODO: Detect version and adapt
    # EGL
    ctx = moderngl.create_context(
        standalone=True, backend="egl", libgl="libGL.so.1", libegl="libEGL.so.1"
    )

    renderer = Renderer(
        ctx,
        fragment_shader=fragment_shader.read(),
        resolution=(width, height),
        duration=duration,
    )

    # Main render loop
    fbo = ctx.framebuffer(color_attachments=[ctx.texture((width, height), 4)])
    fbo.use()
    with imageio.get_writer(output, fps=fps, quality=quality) as writer:
        for frame in tqdm(range(int(duration * fps) - 1)):
            renderer.render(frame / fps)

            # Write video frame
            data = fbo.read(components=3)
            image = Image.frombytes("RGB", fbo.size, data)
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            writer.append_data(np.array(image))
    logging.info("Complete")
