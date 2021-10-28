from io import StringIO
from pathlib import Path

from click.testing import CliRunner

from paraphernalia import glsl

SIMPLE_SHADER = """
#ifdef GL_ES
precision mediump float;
#endif

uniform float u_time;

void main() {
	gl_FragColor = vec4(1.0,0.0,1.0,1.0);
}
"""


def test_render():
    runner = CliRunner()

    with runner.isolated_filesystem():
        shader_path = Path("simple.frag")
        output_path = Path("output.mp4")

        with open(shader_path, "w") as f:
            f.write(SIMPLE_SHADER)

        result = runner.invoke(
            glsl.render,
            [
                str(shader_path),
                "--output",
                str(output_path),
                "--height",
                100,
                "--width",
                100,
                "--duration",
                "1",
            ],
        )
        assert result.exit_code == 0

        assert output_path.exists()
        assert output_path.stat().st_size > 0
