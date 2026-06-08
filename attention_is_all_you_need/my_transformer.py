import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        i = torch.arange(0, d_model, 2).float()
        scaled = i * (-math.log(10000.0) / d_model)
        div_term = torch.exp(scaled)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, dropout: nn.Dropout, mask=None):
        d_k = query.shape[-1]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = scores.softmax(dim=-1)
        if dropout is not None:
            attn = dropout(attn)
        return torch.matmul(attn, value), attn

    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, _ = q.size()

        # linear projections
        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)

        seq_len_q=q.size(1)
        seq_len_k=k.size(1)
        seq_len_v=k.size(1)
        # split into h heads, then transpose
        Q = Q.view(batch_size, seq_len_q, self.h, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.h, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_v, self.h, self.d_k).transpose(1, 2)

        # apply attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(Q, K, V, self.dropout, mask)

        # concat heads and final linear
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.h * self.d_k)
        return self.dropout(self.w_o(x))


class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.ones(features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x_hat = (x - mean) / (std + self.eps)
        return self.gamma * x_hat + self.beta


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([
            ResidualConnection(features, dropout) for _ in range(2)
        ])

    def forward(self, x, src_mask=None):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([
            ResidualConnection(features, dropout) for _ in range(3)
        ])

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.proj(x)


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder,
                 src_embed: InputEmbeddings, tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask=None):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor,
               tgt: torch.Tensor, tgt_mask: torch.Tensor):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)

    @staticmethod
    def build_transformer(src_vocab_size: int, tgt_vocab_size: int,
                          src_seq_len: int, tgt_seq_len: int,
                          d_model: int = 512, N: int = 6, h: int = 8,
                          dropout: float = 0.1, d_ff: int = 2048) -> "Transformer":
        src_embed = InputEmbeddings(src_vocab_size, d_model)
        tgt_embed = InputEmbeddings(tgt_vocab_size, d_model)

        src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
        tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

        encoder_blocks = []
        for _ in range(N):
            enc_attn = MultiHeadAttentionBlock(d_model, h, dropout)
            ff = FeedForwardBlock(d_model, d_ff, dropout)
            encoder_blocks.append(EncoderBlock(d_model, enc_attn, ff, dropout))

        decoder_blocks = []
        for _ in range(N):
            dec_self = MultiHeadAttentionBlock(d_model, h, dropout)
            dec_cross = MultiHeadAttentionBlock(d_model, h, dropout)
            ff = FeedForwardBlock(d_model, d_ff, dropout)
            decoder_blocks.append(DecoderBlock(d_model, dec_self, dec_cross, ff, dropout))

        encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
        decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
        projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

        transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return transformer
