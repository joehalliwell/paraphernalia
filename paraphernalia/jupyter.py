"""
Tools for notebook-based work
"""
import time
from IPython.display import display, Audio, Javascript, HTML

# Audio routines from https://stackoverflow.com/questions/17323336/automatically-play-sound-in-ipython-notebook

# TODO: Assign credit for samples!
DEFAULT_DING = "https://freesound.org/data/previews/80/80921_1022651-lq.ogg"
COMPLETE_DING = "https://freesound.org/data/previews/122/122255_1074082-lq.mp3"


class InvisibleAudio(Audio):
    def _repr_html_(self):
        audio = super()._repr_html_()
        html = f'<div style="display:none">{audio}</div>'
        return html


def ding(url=DEFAULT_DING):
    url = url.replace("'", r"\'")
    widget = f"""
    (function() {{
        var audio = new Audio('{url}');
        audio.play();
    }})();
    """
    handle = display(Javascript(widget), display_id=True)
    time.sleep(0.1)
    handle.update(Javascript(""))


def say(text):
    # Escape single quotes
    text = text.replace("'", r"\'")
    widget = f"""
    (function() {{
        if (window.speechSynthesis) {{
            var synth = window.speechSynthesis;
            synth.speak(new window.SpeechSynthesisUtterance('{text}'));
        }}
    }})();
    """
    handle = display(Javascript(widget), display_id=True)
    time.sleep(0.1)
    handle.update(Javascript(""))
