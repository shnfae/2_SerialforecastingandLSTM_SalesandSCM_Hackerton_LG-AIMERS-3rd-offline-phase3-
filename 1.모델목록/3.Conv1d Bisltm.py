class ConvBiLSTMNet(nn.Module):
    def __init__(self, input_size=7, hidden_size=512, output_size=CFG['PREDICT_SIZE']):
        super(ConvBiLSTMNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.actv = nn.GELU()
        self.dropout = nn.Dropout()
        self.hidden_size = hidden_size

        # BiLSTM layer
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        # Convolutional layers
        # Ensure input shape is (batch, channels, sequence_length)
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.actv(self.conv1(x))

        # Transpose back to (batch, sequence_length, channels) for LSTM
        x = x.permute(0, 2, 1)

        # BiLSTM layer
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]

        # Fully connected layers
        x = self.fc(last_output)
        return x.squeeze(1)
