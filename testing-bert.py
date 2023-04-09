import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from UserIntent import UserIntent


# create lists to hold queries and labels
queries = []
labels = []


# define hyperparameters
MAX_LENGTH = 50
BATCH_SIZE = 16
NUM_EPOCHS = 6
LEARNING_RATE = 2e-5



# load the saved model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                      num_labels=8,
                                                      output_attentions=False,
                                                      output_hidden_states=False)
model.load_state_dict(torch.load('C:\\Users\\rwlra\\OneDrive - Lambton College\\Lambton\\TERM - 2\\Final - Chatbot\\flight_chatbot_model.pth'))
model.eval()

# define the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# get the user input
# user_input = "can i get flight details based on my flight number?"

user_input = "What's the gate number for flight CX789 from Hong Kong to Sydney?"


# tokenize the user input
input_ids = tokenizer.encode(user_input, add_special_tokens=True, max_length=MAX_LENGTH, truncation=True, padding='max_length', return_tensors='pt')

# make a prediction with the trained model
with torch.no_grad():
    outputs = model(input_ids)
    prediction = torch.argmax(outputs[0]).item()
    

print(prediction)

# get the intent label from the dictionary
intent_label = list(UserIntent.label_dict.keys())[list(UserIntent.label_dict.values()).index(prediction)]
    
# print the predicted intent
print(intent_label)
