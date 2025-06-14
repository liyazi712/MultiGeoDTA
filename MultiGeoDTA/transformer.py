import torch
import torch.nn as nn

class TransformerEnc(nn.Module):
    def __init__(self, num_layers=8, d_model=None, nhead=8, dim_feedforward=256,
                 dropout=0.1):
        super(TransformerEnc, self).__init__()

        # Create a TransformerEncoderLayer instance
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        # Create a TransformerEncoder instance
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, src):
        # Pass the input through the Transformer encoder
        output = self.transformer_encoder(src)
        return output

# embedding_dim = 8
# transformer_enc = TransformerEnc(d_model=embedding_dim)
# input_seq = torch.rand(10, 2, embedding_dim)  # (sequence_length, batch_size, embedding_dim)
# output = transformer_enc(input_seq)
#
# print(output.shape)  # 输出的形状应该是 (sequence_length, batch_size, d_model)