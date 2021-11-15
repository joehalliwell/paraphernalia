"""
- Review images in a tree
- Scaling
- KEEP or LOSE
- Back and forward navigation
- Move KEEP to a target folder (rename as necessary...) "kept"
- Move reviewed folders to a target folder "lost"
- "kept" and "lost" should excluded
- Notebook mode?
- Preserve directory structure when losing

TODO:
- Use click
- Move fully processed directories to lose
- Add filename filtering

"""

import imghdr
import logging
import os
import random
from collections import Counter
from pathlib import Path

import click
import moderngl
import moderngl_window as mglw
import PIL
from moderngl_window import geometry, resources
from moderngl_window.text.bitmapped import TextWriter2D
from tqdm import tqdm

import paraphernalia as pa

_LOG = logging.getLogger(__name__)


DEFAULT_KEEP_DIR_NAME = "keep"
DEFAULT_LOSE_DIR_NAME = "lose"

VERTEX_SHADER = """
#version 330

in vec3 in_position;
in vec2 in_texcoord_0;
out vec2 uv0;

void main() {
    gl_Position = vec4(in_position, 1);
    uv0 = in_texcoord_0;
}
"""

FRAGMENT_SHADER = """
#version 330

out vec4 fragColor;
uniform sampler2D texture0;
in vec2 uv0;

void main() {
    fragColor = texture(texture0, uv0);
}
"""


class ReviewApp:
    def __init__(self, review, fullscreen=True):
        """
        The review application
        """
        self.review = review
        self.scale = True
        self.img = None

        # mglw setup
        mglw.settings.WINDOW.update(
            {
                "title": "Review",
                "fullscreen": fullscreen,
                "size": (800, 600),
                "aspect_ratio": None,
                "resizable": True,
            }
        )
        self.window = None

    def run(self):
        """Main event loop. Blocking."""
        self.window = mglw.create_window_from_settings()
        self.window.key_event_func = self.key_event

        self.prog = self.window.ctx.program(
            vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER
        )
        self.writer = TextWriter2D()

        self.load()
        while not self.window.is_closing:
            if Path(self.img.filename) != self.review.path:
                self.load()
            self.render()
            if not self.window.is_closing:
                self.window.swap_buffers()
        self.window.destroy()

    def load(self):
        """Load and display the specified image"""
        # Load the image with PIL
        _LOG.info(f"Loading {self.review.path}")
        img = PIL.Image.open(self.review.path)
        self.img = img

        # Prep for display
        img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        img = img.convert("RGBA")

        # Convert to a texture for display
        self.texture = self.window.ctx.texture(img.size, 4, img.tobytes())
        self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

    def render(self):
        """Render callback"""
        # Setup for rendering
        self.window.use()
        self.window.ctx.clear()
        self.texture.use(location=0)
        self.prog["texture0"].value = 0

        # Compute optimum scaling
        w, h = self.img.size
        tw, th = self.window.viewport_size
        scale = min(tw / w, th / h) if self.scale else 1.0
        scale *= 2
        size = (
            scale * w / tw,
            scale * h / th,
        )

        quad = geometry.quad_2d(size=size)
        quad.render(self.prog)

        todo, keep, lose = self.review.progress
        done = keep + lose

        self.print(
            [
                f"{self.review.index+1:04d}/{self.review.total:04d} [{self.review.verdict:4s}] {self.review.path} {self.img.size[0]}x{self.img.size[1]}",
                f" Progress: {(done/self.review.total*100):.02f}%",
                f"   [T]odo: {todo}",
                f"   [K]eep: {keep}",
                f"   [L]ose: {lose}",
                f"  [S]cale: {self.scale}",
            ]
        )

    def print(self, lines):
        font_size = 24
        x = font_size // 2
        y = font_size // 2
        for line in lines:
            self.writer.text = line
            self.writer.draw((x, self.window.viewport_height - y), size=font_size)
            y += font_size

    def key_event(self, key, action, modifiers):
        """Key handler"""
        keys = self.window.keys
        if action != keys.ACTION_PRESS:
            return

        if key == keys.F:
            self.window.fullscreen = not self.window.fullscreen
            if not self.window.fullscreen:
                self.window.size = (800, 600)
            return
        elif key == keys.S:
            self.scale = not self.scale
            return

        elif key == keys.T:
            self.review.clear()
            self.review.next()
        elif key == keys.LEFT:
            self.review.previous(verdict=None)
        elif key == keys.RIGHT:
            self.review.next(verdict=TODO)
        elif key == keys.K:
            self.review.keep()
            self.review.next()
        elif key == keys.L:
            self.review.lose()
            self.review.next()


TODO = "TODO"
KEEP = "KEEP"
LOSE = "LOSE"
ALL = "ALL"


class Review:
    """Core review logic"""

    def __init__(
        self,
        source_path: str,
        keep_path: str = None,
        lose_path: str = None,
        dryrun: bool = False,
    ):
        self.source_path = Path(source_path).absolute()
        if not self.source_path.exists:
            raise ValueError(f"Source path {self.source_path} does not exist")
        self.keep_path = Path(
            keep_path or self.source_path.parent / DEFAULT_KEEP_DIR_NAME
        ).absolute()
        self.lose_path = Path(
            lose_path or self.keep_path.parent / DEFAULT_LOSE_DIR_NAME
        ).absolute()
        self.dryrun = dryrun

        # Check all paths are different
        assert self.keep_path != self.source_path
        assert self.lose_path != self.source_path
        assert self.keep_path != self.lose_path

        todo = self._todo(self.source_path)
        _LOG.info(f"Source path: {self.source_path} ({len(todo)} files to review)")
        _LOG.info(f"  Keep path: {self.keep_path}")
        _LOG.info(f"  Lose path: {self.lose_path}")
        _LOG.info(f"     Dryrun: {self.dryrun}")

        self.index = 0
        self.total = len(todo)
        self.todo = todo
        self.verdicts = [TODO for _ in range(self.total)]

    @property
    def path(self) -> Path:
        """The current path to review"""
        return self.source_path / self.todo[self.index]

    @property
    def verdict(self):
        """The verdict on the current path"""
        return self.verdicts[self.index]

    @verdict.setter
    def verdict(self, value):
        """Set the verdict on the current path"""
        _LOG.info(f"{self.path} = {value}")
        self.verdicts[self.index] = value

    def scan(self, direction, verdict=None):
        """
        Scan to the next item matching verdict (None matches any)

        Args:
            direction (int): 1 for forward, -1 for backward
            verdict (Optional[int], optional): Status to match. Defaults to None
              which matches everything.
        """
        if direction not in {-1, 1}:
            raise ValueError(f"Invalid direction {direction}")
        for i in range(self.total):
            self.index = (self.index + direction + direction * i) % self.total
            if self.verdict == verdict or verdict is None:
                break

    def next(self, verdict=None):
        """Switch to next object in the sequence, subject to the filters"""
        self.scan(1, verdict)

    def previous(self, verdict=None):
        """Switch to previous object in the sequence, subject to the filters"""
        self.scan(-1, verdict)

    def clear(self):
        """Clear the verdict on the current object"""
        self.verdict = TODO

    def keep(self):
        """Mark the current object to be kept"""
        self.verdict = KEEP

    def lose(self):
        """Mark the current object to be removed"""
        self.verdict = LOSE

    @property
    def progress(self):
        counter = Counter(self.verdicts)
        return counter[TODO], counter[KEEP], counter[LOSE]

    def _todo(self, source):
        """Find candidates within a parent path"""
        files = [
            Path(parent) / file
            for (parent, _dirs, files) in os.walk(source)
            for file in files
        ]
        todo = []
        for f in tqdm(files):
            if imghdr.what(f) in ["jpeg", "png", "tiff"]:
                todo.append(f.relative_to(source))
        return todo

    def _move(self, src: Path, dst: Path):
        """Safely move a file from src to dst"""
        assert src.exists()
        assert src.is_file()
        assert not dst.exists()

        _LOG.info(f"Moving {src} to {dst}")

        if self.dryrun:
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.rename(src, dst)

    def commit(self):
        """Move any empty source directories to lose"""
        _LOG.info(f"Committing {self.progress} changes")
        for (path, verdict) in zip(self.todo, self.verdicts):
            if verdict is TODO:
                continue
            else:
                dst = (
                    self.lose_path / path
                    if verdict == LOSE
                    else self.keep_path / path.name
                )
                self._move(self.source_path / path, dst)

        # TODO: Remove any now empty dirs
        _LOG.info("Cleaning up source directory")


@click.command()
@click.argument(
    "source",
    default=".",
    type=click.Path(exists=True, file_okay=False, readable=True),
)
@click.option("--dryrun", help="Don't move files around", default=False, is_flag=True)
def review(source, dryrun):
    """
    Review images in SOURCE.
    """
    pa.setup_logging()
    review = Review(source_path=source, dryrun=dryrun)
    if review.total == 0:
        _LOG.info("Nothing to review. Exiting")
        return 0
    app = ReviewApp(review)
    try:
        app.run()
    except Exception as e:
        _LOG.exception(e)
    review.commit()
