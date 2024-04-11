
# 6. Bilstm 정의 및 train, validation 관련 정의
class LSTMEncoder(nn.Module):
    def __init__(self, input_size=5, hidden_size=512, latent_dim=128):
        super(LSTMEncoder, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.actv = nn.ReLU()
        self.dropout = nn.Dropout()

        # BiLSTM layer to produce the latent context vector
        self.lstm = nn.LSTM(hidden_size, latent_dim, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = x.float()

        x = x.permute(0, 2, 1)
        x = self.actv(self.bn1(self.conv1(x)))

        x = x.permute(0, 2, 1)
        outputs, (hidden, cell) = self.lstm(x)

        latent_representation = torch.cat((hidden[-1], hidden[-2]), dim=1)
        return latent_representation


class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim=128, hidden_size=512, output_size=1, sequence_length=CFG['TRAIN_WINDOW_SIZE']):
        super(LSTMDecoder, self).__init__()

        # Initial dense layer to reshape latent dim to expected LSTM input
        self.fc = nn.Linear(2 * latent_dim, hidden_size * sequence_length)

        # LSTM layer
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        # Dense layer to produce the final sequence
        self.fc_out = nn.Linear(hidden_size * sequence_length, sequence_length)

    def forward(self, x):
        x = x.float()

        x_transformed = self.fc(x).view(x.size(0), CFG['TRAIN_WINDOW_SIZE'], 512)

        x, _ = self.lstm(x_transformed)

        final_output = self.fc_out(x.reshape(x.size(0), -1))

        return final_output


class biLSTMAutoencoder(nn.Module):
    def __init__(self, input_size=5, hidden_size=512, latent_dim=128, sequence_length=CFG['TRAIN_WINDOW_SIZE']):
        super(biLSTMAutoencoder, self).__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size, latent_dim)
        self.decoder = LSTMDecoder(latent_dim, hidden_size, input_size, sequence_length)

    def forward(self, x):
        x = x.float()

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded
    

model = biLSTMAutoencoder()
