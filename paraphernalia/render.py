import click
import imageio
import moderngl
import numpy as np
from PIL import Image
from tqdm import tqdm


class FakeUniform:
    value = None


VERTEX_SHADER = """
    #version 330
    in vec2 vx;
    out vec2 uv;

    void main() {
        gl_Position = vec4(vx, 0.0, 1.0);
        uv = vx;
    }
    """


@click.command()
@click.argument("fragment_shader", type=click.File("r"))
@click.option("--fps", type=float, default=25.0)
@click.option("--duration", type=float, default=30.0)
@click.option("--width", type=int, default=1024)
@click.option("--height", type=int, default=1024)
@click.option("--quality", type=int, default=6)
def main(fragment_shader, fps, duration, width, height, quality):

    # TODO: Detect version and adapt

    # EGL
    ctx = moderngl.create_context(
        standalone=True,
        backend="egl",
        libgl="libGL.so.1",
        libegl="libEGL.so.1",
    )

    prog = ctx.program(
        vertex_shader=VERTEX_SHADER, fragment_shader=fragment_shader.read()
    )

    # A simple plane to draw on
    vertices = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0])
    vbo = ctx.buffer(vertices.astype("f4"))
    vao = ctx.simple_vertex_array(prog, vbo, "vx")

    # Uniforms
    u_time = prog.get("u_time", FakeUniform())
    u_time.value = 0

    u_resolution = prog.get("u_resolution", FakeUniform())
    u_resolution.value = (width, height)

    u_mouse = prog.get("u_mouse", FakeUniform())
    u_mouse.value = (0, 0)

    # Main render loop
    fbo = ctx.framebuffer(color_attachments=[ctx.texture((width, height), 4)])
    fbo.use()
    with imageio.get_writer("output.mp4", fps=fps, quality=quality) as writer:
        for frame in tqdm(range(duration * fps - 1)):
            u_time.value = frame / fps
            ctx.clear(1.0, 1.0, 1.0)
            vao.render(moderngl.TRIANGLE_STRIP)

            # Write video frame
            data = fbo.read(components=3)
            image = Image.frombytes("RGB", fbo.size, data)
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            writer.append_data(np.array(image))


if __name__ == "__main__":
    main()
