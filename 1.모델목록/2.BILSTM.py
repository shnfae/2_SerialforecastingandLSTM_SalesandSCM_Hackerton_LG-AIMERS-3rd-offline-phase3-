class BiLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=512, output_size=CFG['PREDICT_SIZE']):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True,bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size, output_size)
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        x = x.float()
        h_0 = torch.zeros(2, x.size(0), self.hidden_size, device=device)
        c_0 = torch.zeros(2, x.size(0), self.hidden_size,  device=device)
        lstm_out, _ = self.lstm(x, (h_0,c_0))
        last_output = lstm_out[:, -1, :]
        output = self.relu(self.fc(last_output))
        return output.squeeze(1)
