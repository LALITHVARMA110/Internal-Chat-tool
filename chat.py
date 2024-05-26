import random
import json

import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents from JSON file
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Load trained model and related data
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Initialize and load model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

# Function to get response from the model
def get_response(msg):
    # Tokenize the input message
    sentence = tokenize(msg)
    # Convert input to bag of words
    X = bag_of_words(sentence, all_words)
    # Reshape and convert to tensor
    X = torch.from_numpy(X).to(device).unsqueeze(0)

    # Get model prediction
    output = model(X)
    _, predicted = torch.max(output, dim=1)

    # Get associated tag and probability
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # Check if probability is above threshold, otherwise return default response
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    else:
        return "I am not sure how to respond to that."

# Main block to run the chatbot
if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # Get user input
        sentence = input("You: ")
        if sentence.lower() == "quit":
            break

        # Get and print bot response
        resp = get_response(sentence)
        print(bot_name + ":", resp)