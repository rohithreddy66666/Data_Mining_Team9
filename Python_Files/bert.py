
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW, BertModel, BertTokenizer

# Define the path to your dataset
data_path = '../training_set_rel3.tsv'

# Load the dataset with a different encoding
df = pd.read_csv(data_path, delimiter='\t', encoding='ISO-8859-1')

# Preprocess text data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_texts(texts, tokenizer, max_len=512):
    encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=max_len, return_tensors='pt')
    return encodings['input_ids'], encodings['attention_mask']

input_ids, attention_mask = preprocess_texts(df['essay'], tokenizer)


labels = torch.tensor(df['domain1_score'].values)

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(input_ids, labels, test_size=0.3, random_state=42)
train_mask, temp_mask = train_test_split(attention_mask, test_size=0.3, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
val_mask, test_mask = train_test_split(temp_mask, test_size=0.5, random_state=42)


class BERTRegressor(nn.Module):
    def __init__(self, dropout=0.3):
        super(BERTRegressor, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.bert.config.hidden_size, 256)
        self.relu = nn.ReLU()
        self.out = nn.Linear(256, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_token)
        x = self.relu(self.linear(x))
        x = self.dropout(x)
        x = self.out(x)
        return x

# Initialize the model
model = BERTRegressor()

# Create DataLoader
train_data = TensorDataset(X_train, train_mask, y_train)
val_data = TensorDataset(X_val, val_mask, y_val)
test_data = TensorDataset(X_test, test_mask, y_test)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16)
test_loader = DataLoader(test_data, batch_size=16)

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=3e-5)
loss_fn = nn.MSELoss()

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def train(model, train_loader, val_loader, optimizer, loss_fn, epochs=10):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, labels = [item.to(device) for item in batch]
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [item.to(device) for item in batch]
                outputs = model(input_ids, attention_mask)
                loss = loss_fn(outputs.squeeze(), labels.float())
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}")

train(model, train_loader, val_loader, optimizer, loss_fn, epochs=10)


# Save the model
torch.save(model.state_dict(), '../bert_model.pth')



# Evaluation function
def evaluate(model, data_loader):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels = [item.to(device) for item in batch]
            outputs = model(input_ids, attention_mask)
            predictions.extend(outputs.squeeze().cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return predictions, true_labels

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, np.round(y_pred).astype(int), weights='quadratic')
    return mse, rmse, mae, kappa

def print_metrics(set_name, mse, rmse, mae, kappa):
    print(f"{set_name} set evaluation:")
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Cohen\'s Kappa Score: {kappa}')
    print()

# Ensure GPU usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Evaluate on training set
y_train_pred, y_train_true = evaluate(model, train_loader)
train_mse, train_rmse, train_mae, train_kappa = calculate_metrics(y_train_true, y_train_pred)
print_metrics("Training", train_mse, train_rmse, train_mae, train_kappa)

# Evaluate on validation set
y_val_pred, y_val_true = evaluate(model, val_loader)
val_mse, val_rmse, val_mae, val_kappa = calculate_metrics(y_val_true, y_val_pred)
print_metrics("Validation", val_mse, val_rmse, val_mae, val_kappa)

# Evaluate on test set
y_test_pred, y_test_true = evaluate(model, test_loader)
test_mse, test_rmse, test_mae, test_kappa = calculate_metrics(y_test_true, y_test_pred)
print_metrics("Test", test_mse, test_rmse, test_mae, test_kappa)

