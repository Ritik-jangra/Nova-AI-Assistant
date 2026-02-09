import pyautogui as ui
from Head.Mouth import speak
import time


def minimize():
    speak("Minimizing the current window, Jerry")
    time.sleep(0.3)
    ui.hotkey('win', 'down')

