class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.actv = nn.GELU()

    def forward(self, x):
        out = self.actv(self.conv(x))
        out = torch.cat([x, out], 1)  # Concatenate along the channel dimension
        return out

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.actv = nn.GELU()

    def forward(self, x):
        return self.actv(self.conv(x))

class ConvDenseNetBiLSTMNet(nn.Module):
    def __init__(self, input_size=5, hidden_size=512, output_size=CFG['PREDICT_SIZE'], growth_rate=32, num_layers=4):
        super(ConvDenseNetBiLSTMNet, self).__init__()

        layers = [DenseLayer(input_size + i * growth_rate, growth_rate) for i in range(num_layers)]
        self.dense_block = nn.Sequential(*layers)
        self.transition = TransitionLayer(input_size + num_layers * growth_rate, hidden_size)

        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.dense_block(x)
        x = self.transition(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        x = self.fc(last_output)
        return x.squeeze(1)
