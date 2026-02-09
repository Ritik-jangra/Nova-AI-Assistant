import pyautogui as ui
from Head.Mouth import speak
import time

def close():
    speak("Closing the current window, Jerry")
    time.sleep(0.3)
    ui.hotkey('alt', 'f4')
