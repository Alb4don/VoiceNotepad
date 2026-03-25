import logging
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(
            _ROOT / "voice_notepad.log",
            mode="a",
            encoding="utf-8",
        ),
    ],
)

import wx
from src.gui import VoiceNotepadApp

if __name__ == "__main__":
    app = VoiceNotepadApp()
    app.MainLoop()