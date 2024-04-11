
### feature 3개 + 1 drop일경우

class LTSF_NLinear(torch.nn.Module):
    def __init__(self, window_size=CFG['TRAIN_WINDOW_SIZE'], forcast_size=CFG['PREDICT_SIZE'], individual=True, feature_size=7):
        super(LTSF_NLinear, self).__init__()
        self.window_size = window_size
        self.forcast_size = forcast_size
        self.individual = individual
        self.channels = feature_size

        if self.individual:
            self.Linear = torch.nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(torch.nn.Linear(self.window_size, self.forcast_size))
        else:
            self.Linear = torch.nn.Linear(self.window_size, self.forcast_size)

        self.final_layer = torch.nn.Linear(self.channels, 1)  # To select only sales

    def forward(self, x):
        x=x.float()
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last

        if self.individual:
            output = torch.zeros([x.size(0), self.forcast_size, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)

        # This will select only the sales (assuming it's the first feature)
        x = self.final_layer(x)
        adjusted_seq_last = seq_last[:,:,4].unsqueeze(-1).expand(-1, 21, -1)
        x = x + adjusted_seq_last

        return x.squeeze(-1)  # Output shape [1024, 21]
class informerlstm(nn.Module):
    def __init__(self, input_size=7, hidden_size=512, output_size=CFG['PREDICT_SIZE']):
        super(informerlstm, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True,bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size, output_size)
        )
        self.relu = nn.ReLU()

        self.informer = LTSF_NLinear(window_size=CFG['TRAIN_WINDOW_SIZE'], forcast_size=output_size, individual=True, feature_size=input_size)

    def forward(self, x):
        x = x.float()
        h_0 = torch.zeros(2, x.size(0), self.hidden_size, device=x.device)
        c_0 = torch.zeros(2, x.size(0), self.hidden_size, device=x.device)

        lstm_out, _ = self.lstm(x, (h_0, c_0))
        last_output = lstm_out[:, -1, :]
        x = self.informer(x)  # Note: Make sure the shapes are compatible
        x = self.relu(self.fc(last_output))

        return x  # Output shape [batch_size, output_size]
