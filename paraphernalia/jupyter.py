"""
Tools for notebook-based work
"""
import time

from IPython.display import Javascript, display

# TODO: Assign credit for samples!
DEFAULT_DING = "https://freesound.org/data/previews/80/80921_1022651-lq.ogg"
COMPLETE_DING = "https://freesound.org/data/previews/122/122255_1074082-lq.mp3"


def run_once(js, timeout=10000):
    expiry = time.time() * 1000 + timeout
    widget = f"if (Date.now()<={expiry}){{{js.strip()}}}"
    display(Javascript(widget))


def ding(url=DEFAULT_DING):
    url = url.replace("'", r"\'")
    run_once(f"new Audio('{url}').play();")


def say(text):
    # Escape single quotes
    text = str(text).replace("'", r"\'")
    run_once(
        f"""
        if (window.speechSynthesis) {{
            let u = window.SpeechSynthesisUtterance('{text}');
            u.voice = window.speechSynthesis.getVoices().filter(v => v.lang.search("US") != -1)[0] || null;
            u.pitch = 0.8;
            window.speechSynthesis.speak(u);
        }}
        """
    )
