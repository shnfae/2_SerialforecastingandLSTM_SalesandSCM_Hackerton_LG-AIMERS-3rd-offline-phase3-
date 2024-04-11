from torch.nn.utils.rnn import PackedSequence
from typing import *


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

class VariationalDropout(nn.Module):
    def __init__(self, dropout: float, batch_first: Optional[bool]=False):
        super().__init__()
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout <= 0.:
            return x

        is_packed = isinstance(x, PackedSequence)
        if is_packed:
            x, batch_sizes = x
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = x.size(0)

        # Drop same mask across entire sequence
        if self.batch_first:
            m = x.new_empty(max_batch_size, 1, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        else:
            m = x.new_empty(1, max_batch_size, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        x = x.masked_fill(m == 0, 0) / (1 - self.dropout)

        if is_packed:
            return PackedSequence(x, batch_sizes)
        else:
            return x

class informerlstm(nn.Module):
    def __init__(self, input_size=5, hidden_size=512, output_size=CFG['PREDICT_SIZE']):
        super(informerlstm, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size, output_size)
        )
        self.relu = nn.ReLU()

        self.informer = LTSF_NLinear(window_size=CFG['TRAIN_WINDOW_SIZE'], forcast_size=output_size, individual=True, feature_size=input_size)
        self.unit_forget_bias = True
        self.dropoutw = float=0.
        self.input_drop = VariationalDropout(0., batch_first=True)
        self.output_drop = VariationalDropout(0., batch_first=True)

        self._init_weights()

    def _init_weights(self):
        """
        Use orthogonal init for recurrent layers, xavier uniform for input layers
        Bias is 0 except for forget gate
        """
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name and self.unit_forget_bias:
                nn.init.zeros_(param.data)
                param.data[self.hidden_size:2 * self.hidden_size] = 1

    def _drop_weights(self):
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                param.data = torch.nn.functional.dropout(param.data, p=self.dropoutw, training=self.training).contiguous()

    def forward(self, x):
        x = x.float()
        h_0 = torch.zeros(2, x.size(0), self.hidden_size, device=x.device)
        c_0 = torch.zeros(2, x.size(0), self.hidden_size, device=x.device)

        x = self.input_drop(x)
        self._drop_weights()
        lstm_out, _ = self.lstm(x, (h_0, c_0))
        lstm_out = self.output_drop(lstm_out)

        last_output = lstm_out[:, -1, :]
        x = self.informer(x)  # Note: Make sure the shapes are compatible
        x = self.relu(self.fc(last_output))

        return x  # Output shape [batch_size, output_size]

# This is the modified model, which accounts for the number of LSTM layers when initializing the hidden and cell states.
