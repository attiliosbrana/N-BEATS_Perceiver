import torch.nn as nn
import torch as t

class RNNModel(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_layers: int, rnn_type: str):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        
        # Define the RNN layer here, but do not specify whether it is an LSTM or GRU
        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers)
        
        # Define a fully connected layer to transform the output of the RNN into the desired output size
        self.fc = nn.Linear(self.hidden_size, self.output_size)
    
    # def forward(self, input_tensor: t.Tensor, input_lengths: t.Tensor) -> t.Tensor:
    #     # Pass the input tensor through the RNN layer
    #     rnn_output, hidden = self.rnn(input_tensor, input_lengths)
        
    #     # Transform the output of the RNN using the fully connected layer
    #     output = self.fc(rnn_output)
        
    #     return output

    def forward(self, input_tensor: t.Tensor, input_lengths: t.Tensor) -> t.Tensor:
        # Reshape the input tensor to a 2-dimensional tensor with a batch size of 1
        input_tensor = input_tensor.unsqueeze(1)
        
        # Pass the input tensor through the RNN layer
        rnn_output, hidden = self.rnn(input_tensor, input_lengths)
        
        # Transform the output of the RNN using the fully connected layer
        output = self.fc(rnn_output)
        
        return output

    

class LSTMModel(RNNModel):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_layers: int, batch_size: int):
        super().__init__(input_size, output_size, hidden_size, num_layers, rnn_type='LSTM')
        
        # Replace the RNN layer with an LSTM layer
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)
        
        # Initialize the hx and cx states with tensors of the correct shape
        self.hx = t.zeros(self.num_layers, batch_size, self.hidden_size)
        self.cx = t.zeros(self.num_layers, batch_size, self.hidden_size)

    def forward(self, input_tensor: t.Tensor, input_lengths: t.Tensor) -> t.Tensor:
        # Pass the input tensor and the hx and cx states through the LSTM layer
        rnn_output, hidden = self.rnn(input_tensor, (self.hx, self.cx))
        
        # Transform the output of the RNN using the fully connected layer
        output = self.fc(rnn_output)
        
        return output


class GRUModel(RNNModel):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_layers: int):
        super().__init__(input_size, output_size, hidden_size, num_layers, rnn_type='GRU')
        
        # Replace the RNN layer with a GRU layer
        self.rnn = nn.GRU(self.input_size, self.hidden_size, self.num_layers)

