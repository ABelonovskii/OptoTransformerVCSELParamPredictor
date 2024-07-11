import math
import torch
import torch.nn as nn


class TransformerModel(nn.Module):

    def __init__(self, one_hot_size, train_emb_size, nhead, nhid, nlayers, output_size, dropout, hidden_sizes, seq_len):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer

        ninp = one_hot_size + train_emb_size

        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.trainable_embeddings = nn.Parameter(torch.randn(seq_len, train_emb_size))

        # The decoder takes as input the flattened output from the encoder (ninp * seq_len) concatenated with the original input values (seq_len).
        self.decoder_network = self._build_decoder(ninp * seq_len + seq_len, hidden_sizes, output_size, dropout)

        self.max_len = seq_len
        self.init_weights()

    def _build_decoder(self, input_size, hidden_sizes, output_size, dropout_rate):
        layers = []
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, output_size))
        return nn.Sequential(*layers)

    def init_weights(self):
        initrange = 0.01

        for module in self.decoder_network:
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
                module.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, mask=None):
        batch_size = src.size(0)
        mask = self.create_mask(src)
        # Concatenate trainable embeddings with source embeddings for each batch
        embeddings = torch.cat((src, self.trainable_embeddings.expand(batch_size, -1, -1)), -1)
        # Apply positional encoding to the embeddings
        embeddings = self.pos_encoder(embeddings)
        # Pass the encoded embeddings through the transformer encoder
        output = self.transformer_encoder(embeddings, src_key_padding_mask=mask)
        # Extract the original data values from the last dimension of source embeddings
        original_data = src[:, :, -1].unsqueeze(-1)
        # Concatenate the transformer output with the original data along the last dimension
        output = torch.cat((output, original_data), dim=-1)
        # Flatten the output to prepare for the decoder
        output = output.view(batch_size, -1)
        # Pass the flattened output through the decoder network
        output = self.decoder_network(output)

        return output


    def create_mask(self, src):
        """
        A temporary solution to generalize the processing of different kinds of VCSEL structures:
         - Single layer
         - DBR
         - VCSEL
        """
        batch_size, _, emb_size = src.size()
        mask = torch.ones(batch_size, self.max_len, dtype=torch.bool).to(src.device)

        for i in range(batch_size):
            if src[i, 3, :].sum() == 0:
                indices = [0, 1, 2, 16, 17, 18, 29, 30]
            elif src[i, 10, :].sum() == 0:
                indices = list(range(10)) + [29, 30]
            else:
                indices = list(range(self.max_len))

            mask[i, indices] = False

        return mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
