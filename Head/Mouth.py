import asyncio
import threading
import os
import edge_tts
import pygame
import sys
import time
from colorama import Fore, Style, init

voice = "hi-IN-SwaraNeural"

def print_animated_message(message):
    for char in message:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.095)
    print()
    
def remove_file(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Cleanup error: {e}")

def play_audio(file_path):
    try:
        pygame.init()
        pygame.mixer.init()

        sound = pygame.mixer.Sound(file_path)
        pygame.time.delay(150) 
        sound.play()

        while pygame.mixer.get_busy():
            pygame.time.Clock().tick(10)

        pygame.mixer.quit()
        pygame.quit()

    except Exception as e:
        print(f"Audio error: {e}")

async def amain(text, output_file):
    try:
        communicate = edge_tts.Communicate(text, voice, rate="+25%", pitch="+5Hz")
        await communicate.save(output_file)
    except Exception as e:
        print(f"TTS error: {e}")

def speak1(text):
    output_file = os.path.join(os.getcwd(), "speak.mp3")

    asyncio.run(amain(text, output_file))

    play_audio(output_file)

    remove_file(output_file)

# # TEST
# if __name__ == "__main__":
#     speak1("मेरा नाम प्राची है, और मैं रितिक की सबसे अच्छी दोस्त हूं")
#     # speak1("My name is Prachi, and I am Ritik's best friend.")

def speak(text):
    
    t1 = threading.Thread(target=speak1, args=(text,))
    t2 = threading.Thread(target=print_animated_message, args=(f"N.O.V.A: {text}",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
# speak("I am currently using the new TTS system, which is much better than the previous one. I hope you like it.")