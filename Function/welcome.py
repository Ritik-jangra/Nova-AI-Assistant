from Head.Mouth import speak
from Data.dlg_data import dlg
import random

def welcome():
    welcome = random.choice(dlg.welcomedlg)
    speak(welcome)