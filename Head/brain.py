import sys
import time
import webbrowser
import wikipedia
import threading
from Training_Model.model import mind
from Head.Mouth import speak
 
def load_qa_data(file_path):
    qa_dict = {}
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(':')
            if len(parts) != 2:
                continue
            q, a = parts
            qa_dict[q] = a
    return qa_dict

qa_file_path = r'C:\Users\ritik\OneDrive\Desktop\Jarvis\Data\brain_data\qna_data.txt'
qa_dict = load_qa_data(qa_file_path)

def print_animated_message(message):
    for char in message:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.05)
    print()
    
def save_qa_data(file_path, qa_dict):
    with open(file_path, 'w', encoding='utf-8') as f:
        for q, a in qa_dict.items():
            f.write(f"{q}:{a}\n")
            
def wiki_search(prompt):
    search_prompt = prompt.replace("jarvis", "")
    search_prompt = search_prompt.replace("wikipedia", "")
    
    try:
        wiki_summary = wikipedia.summary(search_prompt, sentences=2)
        animated_thread = threading.Thread(target=print_animated_message, args=(wiki_summary,))
        speak_thread = threading.Thread(target=speak, args=(wiki_summary,))
        
        animated_thread.start()
        speak_thread.start()
        
        animated_thread.join()
        speak_thread.join()
        
        # Assuming 'search_prompt' is defined somewhere
        qa_dict[search_prompt] = wiki_summary
        save_qa_data(qa_file_path, qa_dict)
    except wikipedia.exceptions.DisambiguationError as e:
        speak("The topic is ambiguous. Please be more specific.")
        print("DisambiguationError:")
    except wikipedia.exceptions.PageError:
        google_search(prompt)
        
def google_search(query):
    query = query.replace("who is", "")
    query = query.strip()
    
    if query:
        url = f"https://www.google.com/search?q=" + query
        webbrowser.open(url)
        speak(f"I have opened the {query} results on Google for you.")
        print(f"Opened Google search for: {query}")
    else:
        speak("I couldn't understand the search query.")
        print("No valid query to search.")
    
wiki_search("what is the distance to the moon")
google_search("what is the distance to the moon")


# To run this program ----> python -m Head.brain