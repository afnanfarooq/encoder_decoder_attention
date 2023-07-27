import torch
import torch.nn as nn
import torch.nn.functional as F
import create_data
import practice_torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
import numpy as np

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        hidden = hidden.repeat(1, seq_len, 1)
        energy = F.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention_score = self.v(energy).squeeze(2)
        attention_weights = F.softmax(attention_score, dim=1)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        return context_vector, attention_weights

class EncoderDecoderLSTM(nn.Module):
    def __init__(self, input_features, hidden_size, output_features):
        super(EncoderDecoderLSTM, self).__init__()

        # Encoder LSTM layer
        self.encoder_lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Decoder LSTM layer
        self.decoder_lstm = nn.LSTM(
            input_size=output_features,  # Decoder input size is the same as the output features
            hidden_size=hidden_size,
            batch_first=True
        )

        # Attention mechanism
        self.attention = Attention(hidden_size)

        # Output layer to generate the final output sequence
        self.output_layer = nn.Linear(hidden_size, output_features)

    def forward(self, x):
        # Encoder forward pass
        encoder_output, encoder_hidden = self.encoder_lstm(x)

        # Decoder initial hidden state is set to the final hidden state of the encoder
        decoder_hidden = encoder_hidden

        # Initialize the attention context vector with zeros
        context = torch.zeros(x.size(0), 1, self.decoder_lstm.hidden_size, device=x.device)

        # Decoder forward pass with attention
        decoder_outputs = []
        for i in range(x.size(1)):
            decoder_input = x[:, i:i+1, :]
            decoder_output, decoder_hidden = self.decoder_lstm(decoder_input, decoder_hidden)

            # Apply attention mechanism
            context, attention_weights = self.attention(decoder_output, encoder_output)
            decoder_output = torch.cat((decoder_output, context), dim=2)

            # Append the decoder output to the list of outputs
            decoder_outputs.append(decoder_output)

        # Stack all decoder outputs along the time dimension
        decoder_outputs = torch.cat(decoder_outputs, dim=1)

        # Final output sequence from the decoder
        output_sequence = self.output_layer(decoder_outputs)

        return output_sequence

# The rest of the code remains the same...

input_data = np.array(create_data.amplitude_points_list[:4])

input_data=torch.tensor(input_data,dtype=torch.float32)

#output_data = practice_torch.one_hot_encoded
output_data =np.random.randint(0, 1, size=(20, 4736) )#input_features))
output_data=torch.tensor(output_data[:4],dtype=torch.long)
dataset = TensorDataset(input_data, output_data)


# Split data into training and validation sets (adjust the split ratio as needed)
train_ratio = 0.5
#train_size = int(train_ratio * 20)
#train_input, train_output = input_data[:train_size], output_data[:train_size]
#val_input, val_output = input_data[train_size:], output_data[train_size:]
val_size = int(len(dataset)*0.5)
train_size = len(dataset)- int(len(dataset)*train_ratio)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
# Create DataLoader for training and validation
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

#val_dataset = TensorDataset(val_input, val_output)
val_loader = DataLoader(val_dataset, batch_size=2)






# Example usage:
batch_size = 2
sequence_length = 4736
input_features = 12
output_features = 12
# Create the model
model = EncoderDecoderLSTM(input_features, sequence_length, output_features)

# ... Rest of the code ...
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1

for epoch in range(num_epochs):
   
    model.train()
    print('training started')
    total_loss = 0

    for batch_input, batch_output in train_loader:
        optimizer.zero_grad()
        print(f"Epoch {epoch+1}, Training Loss: {total_loss / len(train_loader)}")

        # Forward pass
        predicted_output = model(batch_input)

        # Compute loss
        loss = criterion(predicted_output, batch_output)

        # Backpropagation through time (BPTT)
        loss.backward()

        # Clip gradients to prevent exploding gradients (optional but recommended)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimization
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Training Loss: {total_loss / len(train_loader)}")

# Validation loop
torch.save(model.state_dict(), '/home/hamza/encoder_decoder/model.txt')
model.eval()
val_loss = 0

with torch.no_grad():
    for batch_input, batch_output in val_loader:
        predicted_output = model(batch_input)
        val_loss += criterion(predicted_output, batch_output).item()

print(f"Validation Loss: {val_loss / len(val_loader)}")
