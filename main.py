from flask import Flask,json, render_template, jsonify
import nltk
import os
from flask import render_template, request, redirect
import requests
import json
from config import Config
import spacy
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, AutoModelForSequenceClassification
from UserIntent import UserIntent
import random



#create instance of Flask app
app = Flask(__name__)

app._static_folder = "C:\\Users\\rwlra\\OneDrive - Lambton College\\Lambton\\TERM - 2\\Final - Chatbot\\static"

app.config.from_object(Config)

nlp = spacy.load('en_core_web_sm')


# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# load your trained model
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=8)
model.load_state_dict(torch.load('flight_chatbot_model.pth', map_location=torch.device('cpu')))


# define BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)



# Define a function to tokenize and normalize text
def preprocess(text):
    
    tokens = nltk.word_tokenize(text.lower())
    sent_tokens = nltk.sent_tokenize(text.lower())
 
    return tokens, sent_tokens
   


# Define the function to get flight details

def get_flight_details(**kwargs):

    # url = app.config['API_URL'] + app.config['ACCESS_KEY']
    url = f"http://api.aviationstack.com/v1/flights?access_key={app.config['ACCESS_KEY']}&arr_country=India"
    
    params = {"access_key": app.config['ACCESS_KEY']}


    for key, value in kwargs.items():
        
        if key == 'Origin' or 'Destination':

            response = requests.get(url)
            print(response.status_code)
            print(response.text)
            if response.status_code == 200:
                
                data = response.json()
                print(data)

                return data
        
        elif key == 'Status':
        
            response = requests.get(url, params={"flight_status": key['Status']})
            
            if response.status_code == 200:
                
                data = response.json()
                
                return response
        
        elif key == 'Details':
             
            response = requests.get(url, params={"dep_iata": key['Origin'], "arr_iata": key['Destination']})
        
            if response.status_code == 200:
                
                data = response.json()
                
                return data
            
        elif key == 'Date':
    
            response = requests.get(url, params={"dep_iata": key['Origin'], "arr_iata": key['Destination'], "flight_date": key['Date']})

        
            if response.status_code == 200:
                
                data = response.json()
                
                return data

        elif key == 'Flight_Number':
             
            response = requests.get(url, params={"dep_iata": key['Origin'], "arr_iata": key['Destination']})
        
            if response.status_code == 200:
                
                data = response.json()
                
                return data
        
    return "Sorry, I couldn't find any flights between those places."






# User Intent Using Language Model

def user_intent(user_input):

     # encode the user input
    encoded_input = tokenizer.encode_plus(
        user_input,
        add_special_tokens=True,
        padding='max_length',
        max_length=50,
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    # make a prediction with the model
    with torch.no_grad():
        outputs = model(encoded_input['input_ids'].to(device), 
                        encoded_input['attention_mask'].to(device))
        _, predicted = torch.max(outputs[0], dim=1)
        predicted_label = predicted.item()

     # get the intent label from the dictionary
    intent_label = list(UserIntent.label_dict.keys())[list(UserIntent.label_dict.values()).index(predicted_label)]

    return intent_label


# Define a function to generate a response to user input
def generate_response(user_input):
    
    # Preprocess the user input
    tokenize_user_input, tokenize_sentence = preprocess(user_input)
    
    print(tokenize_sentence)
    if ['hello', 'how are you?', 'can you help me with my flight?', 'I need help!'] in tokenize_sentence:
        
        greeting_messages = ["Hi there! How can I help you today?", "Hello! What can I do for you?", "Hey! How can I assist you?"]

        return random.choice(greeting_messages)
    
    intent_label = user_intent(user_input)

    print("Intent of user from Model:", intent_label)

    # Check if the user input contains a flight number
    if any(token.isdigit() for token in tokenize_user_input):
        flight_number = [token for token in user_input if token.isdigit()][0]
        chatbot_response =  get_flight_details(Flight_Number = flight_number)

        return chatbot_response
    
    # Check if the user input contains origin and destination places
    if any(token in tokenize_user_input for token in ["from", "to"]):
        origin = [tokenize_user_input[i+1] for i, token in enumerate(tokenize_user_input) if token == "from"][0]
        destination = [tokenize_user_input[i+1] for i, token in enumerate(tokenize_user_input) if token == "to"][0]
        chatbot_response =  get_flight_details(Origin = origin, Destiantion = destination)
        return chatbot_response

    if intent_label == "flight_status":
        
        chatbot_response = get_flight_details(flight_number = flight_number)

        return chatbot_response
          
    elif intent_label == "flight_number":
        pass
    
    elif intent_label == "flight_details":
        
        chatbot_response = get_flight_details(flight_number = flight_number)
    
    elif intent_label == "flight_gate_number":
        pass

    elif intent_label == 'thank you':

        goodbye_message = ["You're welcome!", "Glad to help!"]

        return random.choice(goodbye_message) 
   
    # Return the generated response as the chatbot's response
    return "Sorry, I couldn't find any flights."



#decorator 
@app.route("/chat", methods  = ['GET', 'POST'])
def chat():
    if request.method == 'POST':
        user_input = request.form["user_input"]
        print(user_input)
        response = generate_response(user_input)
        print(response)
        return render_template('index.html', response = response)
    else:
        return render_template('index.html')
  

if __name__ == "__main__":
    app.run(debug=True)
