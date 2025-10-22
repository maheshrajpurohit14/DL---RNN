# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## DESIGN STEPS
### STEP 1: 

Import libraries for data processing, visualization, PyTorch, and machine learning operations.

### STEP 2: 

Load training and testing datasets from CSV files containing stock closing prices.

### STEP 3: 

Extract and reshape closing prices for training and testing datasets properly.

### STEP 4: 

Normalize data using MinMaxScaler based only on training dataset values.

### STEP 5: 

Create sequences of 60 previous prices to predict next stock price.

### STEP 6: 

Convert sequences and targets into PyTorch tensors for model training.

### STEP 7: 

Create TensorDataset and DataLoader for batch processing during model training.

### STEP 8: 

Define RNN model with two layers, hidden size 64, and linear output.

### STEP 9: 

Initialize model, move to GPU if available, and define MSE loss.

### STEP 10: 

Train model: forward pass, compute loss, backpropagate, optimize parameters iteratively.

### STEP 11: 

Evaluate model on test set without gradients, predict and inverse transform prices.

### STEP 12:

Plot actual versus predicted stock prices and display last predicted value.

## PROGRAM

### Name: Mahesh Raj Purohit J

### Register Number: 212222240058

```python


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


## Step 1: Load and Preprocess Data
# Load training and test datasets
df_train = pd.read_csv('trainset.csv')
df_test = pd.read_csv('testset.csv')


# Use closing prices
train_prices = df_train['Close'].values.reshape(-1, 1)
test_prices = df_test['Close'].values.reshape(-1, 1)


# Normalize the data based on training set only
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_prices)
scaled_test = scaler.transform(test_prices)


# Create sequences
def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

seq_length = 60
x_train, y_train = create_sequences(scaled_train, seq_length)
x_test, y_test = create_sequences(scaled_test, seq_length)
x_train.shape, y_train.shape, x_test.shape, y_test.shape


# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


# Create dataset and dataloader
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


## Step 2: Define RNN Model
class RNNModel(nn.Module):
  def __init__(self,input_size=1,hidden_size=64,num_layers=2,output_size=1):
    super(RNNModel,self).__init__()
    self.rnn=nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
    self.fc=nn.Linear(hidden_size,output_size)
  def forward(self,x):
    out,_=self.rnn(x)
    out=self.fc(out[:,-1,:])
    return out


model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


!pip install torchinfo


from torchinfo import summary

# input_size = (batch_size, seq_len, input_size)
summary(model, input_size=(64, 60, 1))

criterion =nn.MSELoss()
optimizer =torch.optim.Adam(model.parameters(),lr=0.001)

## Step 3: Train the Model
def train_model(model, train_loader, criterion, optimizer, epochs=20):
  train_losses = []
  model.train()
  for epoch in range(epochs):
    total_loss = 0
    for x_batch, y_batch in train_loader:
      x_batch,y_batch=x_batch.to(device),y_batch.to(device)
      optimizer.zero_grad()
      outputs=model(x_batch)
      loss=criterion(outputs, y_batch)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
    train_losses.append(total_loss/len(train_loader))
    print (f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/ len(train_loader):.4f}')

  print('Name : SAI DARSHINI R S  ')
  print('Reg.No : 212223230178')
  plt.plot(train_losses, label='Training Loss')
  plt.xlabel('Epoch')
  plt.ylabel('MSE Loss')
  plt.title('Training Loss Over Epochs')
  plt.legend()
  plt.show()

train_model (model, train_loader, criterion, optimizer)


## Step 4: Make Predictions on Test Set
model.eval()
with torch.no_grad():
    predicted = model(x_test_tensor.to(device)).cpu().numpy()
    actual = y_test_tensor.cpu().numpy()

# Inverse transform the predictions and actual values
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(actual)

# Plot the predictions vs actual prices
print('Name : SAI DARSHINI R S  ')
print('Reg.No : 212223230178')
plt.figure(figsize=(10, 6))
plt.plot(actual_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Stock Price Prediction using RNN')
plt.legend()
plt.show()
print(f'Predicted Price: {predicted_prices[-1]}')
print(f'Actual Price: {actual_prices[-1]}')



```

### OUTPUT

## Training Loss Over Epochs Plot

<img width="273" height="380" alt="image" src="https://github.com/user-attachments/assets/91fa40b4-c1cf-4227-a974-bb76c20ba3d0" />




## True Stock Price, Predicted Stock Price vs time

<img width="956" height="549" alt="image" src="https://github.com/user-attachments/assets/ce30887e-7439-4e6e-b167-3baf55aab4e3" />


### Predictions
<img width="1086" height="711" alt="image" src="https://github.com/user-attachments/assets/1eff6c17-6859-4b73-bdf2-e505faa75566" />



## RESULT

VGG19 model was fine-tuned and tested successfully. The model achieved good accuracy with correct predictions on sample test images.
