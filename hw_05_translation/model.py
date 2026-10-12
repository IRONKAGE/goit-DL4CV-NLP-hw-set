import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_seq):
        embedded = self.dropout(self.embedding(input_seq))
        outputs, hidden = self.gru(embedded)
        # Об'єднуємо два напрямки Bi-GRU
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        return outputs, hidden

class AttentionDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout=0.1, max_length=35):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, hidden_size)

        # Механізм уваги Багданау (Bahdanau Attention)
        self.attn = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.v = nn.Linear(self.hidden_size, 1, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(self.hidden_size * 3, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size * 4, output_size)

    def forward(self, input_step, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input_step)) # [batch_size, 1, hidden_size]

        # Розрахунок енергії та ваг уваги
        seq_len = encoder_outputs.shape[1]
        hidden_expanded = hidden.unsqueeze(1).repeat(1, seq_len, 1) # [batch_size, seq_len, hidden_size]

        energy = torch.tanh(self.attn(torch.cat((hidden_expanded, encoder_outputs), dim=2)))
        attention_weights = F.softmax(self.v(energy).squeeze(2), dim=1) # [batch_size, seq_len]

        # Context Vector: батч-матричне перемноження
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs) # [batch_size, 1, hidden_size * 2]

        # Прохід через GRU декодера
        rnn_input = torch.cat((embedded, context_vector), dim=2)
        output, hidden = self.gru(rnn_input, hidden.unsqueeze(0))

        # Фінальна класифікація токена
        prediction = self.out(torch.cat((output.squeeze(1), context_vector.squeeze(1), embedded.squeeze(1)), dim=1))

        return prediction, hidden.squeeze(0), attention_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.output_size

        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(source)

        decoder_input = target[:, 0].unsqueeze(1) # SOS_TOKEN

        for t in range(1, target_len):
            output, hidden, _ = self.decoder(decoder_input, hidden, encoder_outputs)
            outputs[:, t, :] = output

            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            decoder_input = target[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)

        return outputs
