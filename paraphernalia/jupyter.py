"""
Tools for notebook-based work
"""
import time
from IPython.display import display, Javascript

# TODO: Assign credit for samples!
DEFAULT_DING = "https://freesound.org/data/previews/80/80921_1022651-lq.ogg"
COMPLETE_DING = "https://freesound.org/data/previews/122/122255_1074082-lq.mp3"


def run_once(js, timeout=10000):
    expiry = time.time() * 1000 + timeout
    widget = f"if (Date.now() <= {expiry}){{{js.strip()}}}"
    display(Javascript(widget))


def ding(url=DEFAULT_DING):
    url = url.replace("'", r"\'")
    run_once(f"new Audio('{url}').play();")


def say(text):
    # Escape single quotes
    text = text.replace("'", r"\'")
    run_once(
        f"""
        if (window.speechSynthesis) {{
            var synth = window.speechSynthesis;
            synth.speak(new window.SpeechSynthesisUtterance('{text}'));
        }}
        """
    )
