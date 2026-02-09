import random
import time
import pyautogui as ui
import webbrowser
import os

from Head.Mouth import speak
from Data.dlg_data.dlg import res_open


def open_app(command):
    app = command.lower().replace("open", "").strip()
    speak(random.choice(res_open))

    ui.press("win")
    time.sleep(0.5)
    ui.write(app)
    time.sleep(0.5)
    ui.press("enter")


def open_website(command):
    site = command.lower().replace("open", "").strip()

    if not site.startswith(("http://", "https://")):
        site = "https://" + site

    speak(random.choice(res_open))
    webbrowser.open(site)


def open_file(command):
    path = command.replace("open", "").strip()

    speak(random.choice(res_open))

    if os.path.exists(path):
        os.startfile(path)
    else:
        ui.press("win")
        time.sleep(0.5)
        ui.write(path)
        time.sleep(0.5)
        ui.press("enter")


def open_command(command):
    command = command.lower().strip()

    # website
    if any(x in command for x in [".com", ".in", ".org", "http"]):
        open_website(command)

    # file / folder
    elif "\\" in command or "/" in command:
        open_file(command)

    # application
    else:
        open_app(command)


# ---------- TEST ----------
if __name__ == "__main__":
    open_command("open youtube.com")
