import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from UserIntent import UserIntent
# define hyperparameters
MAX_LENGTH = 50
BATCH_SIZE = 16
NUM_EPOCHS = 6
LEARNING_RATE = 2e-5

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Load Intense Data
queries = UserIntent.queries
labels  = UserIntent.labels

# initialize the tokenizer and encode the queries
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
encoded_data = tokenizer.batch_encode_plus(
    queries,
    add_special_tokens=True,
    padding='max_length',
    max_length=MAX_LENGTH,
    truncation=True,
    return_attention_mask=True,
    return_tensors='pt'
)

# convert the labels to a tensor
label_dict = {label: i for i, label in enumerate(set(labels))}
labels = [label_dict[label] for label in labels]
labels = torch.tensor(labels)

# create a dataset and dataloader
dataset = torch.utils.data.TensorDataset(encoded_data['input_ids'],
                                          encoded_data['attention_mask'],
                                          labels)
dataloader = torch.utils.data.DataLoader(dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)

# initialize the model and optimizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)

# Set up the loss function and number of training epochs
loss_fn = torch.nn.CrossEntropyLoss()

# train the model
# Fine-tune the BERT model
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for step, batch in enumerate(dataloader):
        b_input_ids = batch[0].to(device)
        b_labels = batch[1].argmax(dim=1).to(device)  # convert one-hot encoded labels to indices
        outputs = model(b_input_ids)
        loss = loss_fn(outputs[0], b_labels)
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.zero_grad()
    avg_train_loss = total_loss / len(dataloader)
    print("Epoch:", epoch+1, "Training Loss:", avg_train_loss)

# Save the trained model
torch.save(model.state_dict(), 'flight_chatbot_model.pth')

# Load the trained model
model.load_state_dict(torch.load('flight_chatbot_model.pth'))
