import random
import time
import pyautogui as gui

from Head.Ear import listen
from Head.Mouth import speak
from Training_Model.model import mind
from Function.welcome import welcome
from Data.dlg_data.dlg import wake_key_word, bye_key_word, res_bye
from Automations.open import open_command
from Automations.close import close
from Automations.minimize import minimize
from Function.check_online_offline import is_online
from friday.Fspeak import fspeak


# startup internet check
if is_online():
    fspeak("Internet connection is active. Nova is ready to assist you.")
else:
    fspeak("Internet connection is not active. Please check your internet connection.")


def Nova():
    while True:
        if not is_online():
            time.sleep(1)
            continue

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
            close()
            
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
            speak("Muting the video")
            gui.press("M")
        
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

        # Nova ai
        elif text.startswith("Nova"):
            clean_text = text.replace("Nova", "").strip()
            reply = mind(clean_text)
            speak(reply)

        # fallback
        else:
            reply = mind(text)
            speak(reply)

if __name__ == "__main__":
    Nova()