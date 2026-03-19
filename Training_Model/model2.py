import json
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the JSON data
with open('Data/brain_data/dta.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
    
# Extract training data
training_data = []
for intent in data.get('intents', []):
    if 'patterns' in intent:
        for pattern in intent['patterns']:
            training_data.append((pattern, intent['tag']))
    else:
        print(f"Warning: 'patterns' key not found in intent: {intent}")
            
#  Check if training data is empty
if not training_data:
    print("Error: No training data found. Please check the JSON file.")
else:
    # Prepare features and labels
    X, y = zip(*training_data)
    
    # Convert text data to numerical format
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X)
    
    # Train the Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(X, y)
    
    def get_response(user_input):
        # Convert user input to numerical format
        user_input_vector = vectorizer.transform([user_input])
        
        # Predict the intent
        predicted_intent = classifier.predict(user_input_vector)[0]
        
        # Get a random response based on the predicted intent
        for intent in data.get('intents', []):
            if intent.get('tag') == predicted_intent:
                responses = intent.get('responses', [])
                if responses:
                    return random.choice(responses)
        return "I'm not sure how to respond to that."
                
while True:
    user_input = input("You: ")
    response = get_response(user_input)
    print(f"Nova: {response}")