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
    filename = "simple.frag"
    with runner.isolated_filesystem() as td:
        with open(filename, "w") as f:
            f.write(SIMPLE_SHADER)

        result = runner.invoke(glsl.render, [filename, "--duration", "5"])
        assert result.exit_code == 0
        output = Path(td) / "output.mp4"

        assert output.exists()
        assert output.stat().st_size > 0
