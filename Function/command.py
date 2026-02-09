import random
import pyautogui as gui
import time

from Automations.open import open_command
from Automations.close import close_command
from Data.dlg_data.dlg import wake_key_word, bye_key_word, res_bye
from Function.welcome import welcome
from Head.Ear import listen
from Head.Mouth import speak
from Training_Model.model import mind


def cmd():

    while True:
        text = listen()
        if not text:
            continue

        text = text.lower().strip()

        # wake
        if text in wake_key_word:
            welcome()

        # bye
        elif text in bye_key_word:
            speak(random.choice(res_bye))
            break

        # open
        elif text.startswith("open"):
            open_command(text)

        # close
        elif text.startswith("close"):
            close_command(text)
            
        # SEARCH 
        elif text.startswith("search"):
            query = text.replace("search", "").strip()

            if not query:
                speak("What should I search?")
                continue

            speak(f"Searching {query}")
            time.sleep(0.3)

            # focus website search bar 
            gui.press("/")          # YouTube, Wikipedia, GitHub etc.
            time.sleep(0.3)

            # CLEAR OLD SEARCH 🔥
            gui.hotkey("ctrl", "a")
            time.sleep(0.1)
            gui.press("backspace")

            # type new search
            gui.typewrite(query, interval=0.05)
            gui.press("enter")

            # wait for results
            time.sleep(2)
            
        
        # minimize
        elif "minimize" in text or "minimise" in text:
            speak("Minimizing the current window")
            time.sleep(0.3)
            gui.hotkey("win", "down")
            
        # maximize
        elif "maximize" in text or "maximise" in text:
            speak("Maximizing the current window")
            time.sleep(0.3)
            gui.hotkey("win", "up")

        # volume up
        elif "volume up" in text:
            speak("Increasing volume")
            gui.press("volumeup")

        # volume down
        elif "volume down" in text:
            speak("Decreasing volume")
            gui.press("volumedown")

        # mute
        elif "mute" in text or "unmute" in text:
            speak("Muting the system")
            gui.press("volumemute")
        
        # play / pause video
        elif "play video" in text or "pause video" in text:
            speak("Play pause video")
            gui.press("k")
            
        # fullscreen
        elif "fullscreen" in text:
            speak("Fullscreen mode")
            gui.press("f")
            
        # next video
        elif "next video" in text:
            speak("Next video")
            gui.hotkey("shift", "n")
        
        # previous video
        elif "previous video" in text:
            speak("Previous video")
            gui.hotkey("shift", "p") 
        
         # captions on/off
        elif "captions" in text or "subtitles" in text:
            speak("Toggling captions")
            gui.press("c")

        # screenshot
        elif "screenshot" in text or "take screenshot" in text:
            speak("Taking screenshot")
            gui.hotkey("win", "prtsc")

        # switch window
        elif "switch window" in text or "change window" in text:
            speak("Switching window")
            gui.hotkey("alt", "tab")

        # scroll down
        elif "scroll down" in text:
            speak("Scrolling down")
            gui.scroll(-500)

        # scroll up
        elif "scroll up" in text:
            speak("Scrolling up")
            gui.scroll(500)

        # jarvis ai
        elif text.startswith("jarvis"):
            clean_text = text.replace("jarvis", "").strip()
            reply = mind(clean_text)
            speak(reply)

        # fallback
        else:
            reply = mind(text)
            speak(reply)


if __name__ == "__main__":
    cmd()
