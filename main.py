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

    url = app.config['API_URL']
    # url = f"http://api.aviationstack.com/v1/flights?access_key={app.config['ACCESS_KEY']}&arr_country=India"
    
    params = {
        "access_key": app.config['ACCESS_KEY'],
        "limit": 3
        }


    for key, value in kwargs.items():
        
        if key == 'Origin' or 'Destination':
            
            params = {
            "access_key": app.config['ACCESS_KEY'],
            "limit": 10,
            "dep_iata": kwargs['Origin'], 
            "arr_iata": kwargs['Destination']
            }
            response = requests.get(url, params=params)

            if response.status_code == 200:
                
                data = response.json()

                return data
        
        elif key == 'Status':
        
            response = requests.get(url, params={"flight_status": kwargs['Status']})
            
            if response.status_code == 200:
                
                data = response.json()
                
                return response
        
        elif key == 'Details':
             
            response = requests.get(url, params={"dep_iata": kwargs['Origin'], "arr_iata": kwargs['Destination']})
        
            if response.status_code == 200:
                
                data = response.json()
                
                return data
            
        elif key == 'Date':
    
            response = requests.get(url, params={"dep_iata": kwargs['Origin'], "arr_iata": kwargs['Destination'], "flight_date": kwargs['Date']})

        
            if response.status_code == 200:
                
                data = response.json()
                
                return data

        elif key == 'Flight_Number':
             
            response = requests.get(url, params={"dep_iata": kwargs['Origin'], "arr_iata": kwargs['Destination']})
        
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
    
    if tokenize_sentence[0] in ['hello', 'how are you?', 'can you help me with my flight?', 'I need help!']:
        
        greeting_messages = ["Hi there! How can I help you today?", "Hello! What can I do for you?", "Hey! How can I assist you?"]

        return {"static" : random.choice(greeting_messages)}
    

    ## Predicting User Intent (Classifying Using Trained Model)
    intent_label = user_intent(user_input)

    print("Intent of user from Model:", intent_label)

    # Check if the user input contains a flight number
    if any(token.isdigit() for token in tokenize_user_input):
        flight_number = [token for token in user_input if token.isdigit()][0]
        chatbot_response =  get_flight_details(Flight_Number = flight_number)

        return {"API" : chatbot_response}
    
    # Check if the user input contains origin and destination places
    if any(token in tokenize_user_input for token in ["from", "to"]):
        origin = [tokenize_user_input[i+1] for i, token in enumerate(tokenize_user_input) if token == "from"][0]
        destination = [tokenize_user_input[i+1] for i, token in enumerate(tokenize_user_input) if token == "to"][0]
        chatbot_response =  get_flight_details(Origin = origin, Destination = destination)
        
        return {"API" : chatbot_response}

    if intent_label == "flight_status":
        
        chatbot_response = get_flight_details(flight_number = flight_number)

        return {"API" : chatbot_response}
          
    elif intent_label == "flight_number":
        pass
    
    elif intent_label == "flight_details":
        
        chatbot_response = get_flight_details(flight_number = flight_number)
    
    elif intent_label == "flight_gate_number":
        pass

    elif intent_label == 'thank you':

        goodbye_message = ["You're welcome!", "Glad to help!"]

        return {"static" : random.choice(goodbye_message)}
   
    # Return the generated response as the chatbot's response
    return {"static" : "Sorry, I couldn't find any flights."}



#decorator 
@app.route("/chat", methods  = ['GET', 'POST'])
def chat():
    if request.method == 'POST':
        
        user_input = request.form["user_input"]
        
        print(user_input)
        
        response = generate_response(user_input)
        
        for key,value in response.items():
            
            if key == 'API':

                res_obj = response['API']['data']
                
                ## Fetching Data from API Response
                data_list = []
                
                for flight_data in res_obj:
                    
                    
                    ## Fetching Data from API Response
                    data = {}
                

                    for key,value in flight_data.items():
                        
                        if key == 'flight_date':
                            data['dates'] = value
                        elif key == 'flight_status':
                            data['status'] = value
                        elif key == 'departure':
                            data['departure'] = flight_data['departure']['airport']    
                        elif key == 'arrival':
                            data['arrival'] = flight_data['arrival']['airport']
                        elif key == 'airline':
                            data['airline'] = flight_data['airline']['name']
                        elif key == 'flight':
                            data['number'] = flight_data['flight']['number']
                    
                    data_list.append(data)
                
                print(data_list)
                
                return jsonify(response=data_list)
            
            else:
            
                return jsonify(response=value)
    
    else:

        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
