import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.attn = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):

        # hidden = [batch size, hid dim]
        # encoder_outputs = [batch size, src len, hid dim]

        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        # hidden = [batch size, src len, hid dim]

        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, src len, hid dim]

        attention = self.v(energy).squeeze(2)

        # attention= [batch size, src len]

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, args, vocab_size, attention):
        super().__init__()

        self.output_dim = vocab_size
        self.attention = attention

        self.embedding = nn.Embedding(vocab_size, args.hidden_dim)

        self.rnn = nn.GRU(args.hidden_dim + args.hidden_dim, args.hidden_dim)

        self.fc_out = nn.Linear(
            args.hidden_dim + args.hidden_dim + args.hidden_dim, vocab_size)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, input, hidden, encoder_outputs):

        # input = [batch size]
        # hidden = [batch size, hid dim]
        # encoder_outputs = [batch size, src len, hid dim]

        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, hidden dim]

        a = self.attention(hidden, encoder_outputs)

        # a = [batch size, src len]

        a = a.unsqueeze(1)

        # a = [batch size, 1, src len]

        weighted = torch.bmm(a, encoder_outputs)

        # weighted = [batch size, 1, hid dim]

        weighted = weighted.permute(1, 0, 2)

        # weighted = [1, batch size, hid dim]

        rnn_input = torch.cat((embedded, weighted), dim=2)

        # rnn_input = [1, batch size, hid dim + hid dim]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # output = [seq len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(
            torch.cat((output, weighted, embedded), dim=1))

        # prediction = [batch size, output dim]

        return prediction, hidden.squeeze(0)
