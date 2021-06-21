"""
Tools for notebook-based work
"""
import time
from IPython.display import display, Audio, Javascript, HTML

# Audio routines from https://stackoverflow.com/questions/17323336/automatically-play-sound-in-ipython-notebook

# TODO: Assign credit for samples!
DEFAULT_DING = "https://freesound.org/data/previews/80/80921_1022651-lq.ogg"
COMPLETE_DING = "https://freesound.org/data/previews/122/122255_1074082-lq.mp3"

# HACK: Wait for a bit for the message to arrive then clean up
CLEANUP_DELAY = 0.2


def ding(url=DEFAULT_DING):
    url = url.replace("'", r"\'")
    widget = f"""
    (function() {{
        var audio = new Audio('{url}');
        audio.play();
    }})();
    """
    handle = display(Javascript(widget), display_id=True)
    time.sleep(CLEANUP_DELAY)
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
    time.sleep(CLEANUP_DELAY)
    handle.update(Javascript(""))
