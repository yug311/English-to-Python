import math
import torch
import torch.nn as nn
from DataPreprocess import english_lang, python_lang


embedding_size = 512
eng_max_seq = 24
py_max_seq = 64
batch_size = 32
num_heads = 8
num_layers = 5
dropout_prob = 0.2
lr = 0.001
epochs = 5

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i/self.d_model)
        position = (torch.arange(self.max_sequence_length)
                          .reshape(self.max_sequence_length, 1))
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)

        if self.d_model % 2 == 1:
          PE = PE[:, 0:-1]
        return PE

class LayerNormalization(nn.Module):
  def __init__(self, d_model=512, eps=1e-5):
    super(LayerNormalization, self).__init__()
    self.d_model = d_model
    self.eps = eps
    self.gamma = nn.Parameter(torch.ones(d_model))
    self.beta = nn.Parameter(torch.zeros(d_model))

  def forward(self, x):
    dims = [-1]
    mean = x.mean(dims, keepdim=True) #30 x 21 x 1
    var = ((x - mean) ** 2).mean(dims, keepdim=True)
    std = (var + self.eps).sqrt()
    y = (x - mean) / std
    out = self.gamma * y + self.beta
    return out

class MultiHeadCrossAttention(nn.Module):
  def __init__(self, d_model=512, num_heads=8):
    super(MultiHeadCrossAttention, self).__init__()
    self.d_model = d_model
    self.num_heads = num_heads
    self.head_dim = d_model // num_heads
    self.q_layer = nn.Linear(d_model, d_model)
    self.kv_layer = nn.Linear(d_model, 2 * d_model)
    self.out_layer = nn.Linear(d_model, d_model)

  def forward(self, x, y, mask=None):
    #x: 32 x 21 x 512     y: 32 x 51 x 512.      32 x 21 x 51
    #1
    batch_size, seq_length, feature_length = x.size()
    kv = self.kv_layer(x) #32 x 21 x 1024
    q = self.q_layer(y) #32 x 51 x 512
    kv = kv.reshape(batch_size, seq_length, self.num_heads, 2 * self.head_dim) #32 x 21 x 8 x 128
    kv = kv.permute(0, 2, 1, 3) #32 x 8 x 21 x 128
    k, v = kv.chunk(2, dim=-1) # each 32 x 8 x 21 x 64
    q = q.reshape(batch_size, 64, self.num_heads, self.head_dim) #32 x 51 x 8 x 64
    q = q.permute(0, 2, 1, 3) #32 x 8 x 51 x 64
    values = self.compute_attention(q, k, v, mask)
    values = values.reshape(batch_size, 64, self.num_heads * self.head_dim) #32 x 21 x 512
    values = self.out_layer(values) # 32 x 21 x 512
    return values # 32 x 21 x 512

  def compute_attention(self, q, k, v, mask=None): # each 32 x 8 x 21 x 64
    d_k = q.size()[-1]
    attention = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)# 32 x 8 x 51 x 21
    if mask is not None:
      attention = attention.permute(1, 0, 2, 3) + mask
      attention = attention.permute(1, 0, 2, 3)
    attention = torch.softmax(attention, dim=-1) #along columns to rows sum to 1
    vals = torch.matmul(attention, v) #32 x 8 x 21 x 64
    return vals


class MultiHeadAttention(nn.Module):
  def __init__(self, d_model=512, num_heads=8):
    super(MultiHeadAttention, self).__init__()
    self.d_model = d_model
    self.num_heads = num_heads
    self.head_dim = d_model // num_heads
    self.qkv_layer = nn.Linear(d_model, 3 * d_model)
    self.out_layer = nn.Linear(d_model, d_model)

  def forward(self, x, mask=None): #32 x 21 x 512
    batch_size, seq_length, feature_length = x.size()
    qkv = self.qkv_layer(x) #32 x 21 x 1536
    qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim) #32 x 21 x 8 x 192
    qkv = qkv.permute(0, 2, 1, 3) #32 x 8 x 21 x 192
    q, k, v = qkv.chunk(3, dim=-1) # each 32 x 8 x 21 x 64
    values = self.compute_attention(q, k, v, mask) # 32 x 8 x 21 x 64
    values = values.reshape(batch_size, seq_length, self.num_heads * self.head_dim) #32 x 21 x 512
    values = self.out_layer(values) # 32 x 21 x 512
    return values # 32 x 21 x 512

  def compute_attention(self, q, k, v, mask=None): # each 32 x 8 x 21 x 64
    d_k = q.size()[-1]
    attention = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)# 32 x 8 x 21 x 21
    if mask is not None:
      attention = attention.permute(1, 0, 2, 3) + mask
      attention = attention.permute(1, 0, 2, 3)
    attention = torch.softmax(attention, dim=-1) #along columns to rows sum to 1
    vals = torch.matmul(attention, v) #32 x 8 x 21 x 64
    return vals


class FeedForward(nn.Module):
  def __init__(self, d_model=512, hidden=2048):
    super(FeedForward, self).__init__()
    self.d_model = d_model
    self.hidden = hidden
    self.l1 = nn.Linear(self.d_model, self.hidden)
    self.reLU = nn.ReLU()
    self.dropout = nn.Dropout(p=0.1)
    self.l2 = nn.Linear(self.hidden, self.d_model)

  def forward(self, x):
    x = self.l1(x)
    x = self.reLU(x)
    x = self.dropout(x)
    x = self.l2(x)
    return x

class EncoderLayer(nn.Module):
  def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
    super(EncoderLayer, self).__init__()

    self.attention = MultiHeadAttention()
    self.norm1 = LayerNormalization()
    self.dropout1 = nn.Dropout(p=0.1)
    self.ffn = FeedForward()
    self.norm2 = LayerNormalization()
    self.dropout2 = nn.Dropout(p=0.1)


    #add some dropout layers

  def forward(self, x, mask):
    residual_x = x
    x = self.attention(x, mask=mask)
    x = self.dropout1(x)
    x = self.norm1(x + residual_x)
    residual_x = x
    x = self.ffn(x)
    x = self.dropout2(x)
    x = self.norm1(x + residual_x)
    return x

class SequentialEncoder(nn.Sequential):
  def forward(self, *inputs):
      x, mask = inputs
      for module in self._modules.values():
          y = module(x, mask) #30 x 51 x 512
      return y

class Encoder(nn.Module):
  def __init__(self, input_size=english_lang.n_words, d_model=512, ffn_hidden=2048, num_heads=8, drop_prob=0.2, num_layers=5):
    super(Encoder, self).__init__()
    self.embedding = nn.Embedding(input_size, d_model)
    #not adding positional encodings yet
    self.layer = SequentialEncoder(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])
    self.d_model = d_model

  def forward(self, input, mask):
    embedded = self.embedding(input) #32 x 21 x 512
    batch_size = len(embedded)
    positional = PositionalEncoding(self.d_model, 24) #21 x 512
    positional_encodings = positional().unsqueeze(0).repeat(batch_size, 1, 1)
    # original_tensor.unsqueeze(0).repeat(6, 1, 1)
    return self.layer(embedded + positional_encodings, mask)

class DecoderLayer(nn.Module):
  def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
      super(DecoderLayer, self).__init__()
      self.self_attention = MultiHeadAttention()
      self.norm1 = LayerNormalization()
      self.dropout1 = nn.Dropout(p=0.1)
      self.encoder_decoder_attention = MultiHeadCrossAttention()
      self.norm2 = LayerNormalization()
      self.dropout2 = nn.Dropout(p=0.1)
      self.ffn = FeedForward()
      self.norm3 = LayerNormalization()
      self.dropout3 = nn.Dropout(p=0.1)

  def forward(self, x, y, mask, mask2):

    resid_y = y
    y = self.self_attention(y, mask)
    y = self.dropout1(y)
    y = self.norm1(y + resid_y)

    resid_y = y
    y = self.encoder_decoder_attention(x, y, mask2)
    y = self.dropout2(y)
    y = self.norm2(y + resid_y)

    resid_y = y
    y = self.ffn(y)
    y = self.dropout3(y)
    y = self.norm3(y + resid_y)
    return y


class SequentialDecoder(nn.Sequential):
  def forward(self, *inputs):
      x, y, mask, mask2 = inputs
      for module in self._modules.values():
          y = module(x, y, mask, mask2) #30 x 51 x 512
      return y

class Decoder(nn.Module):
  def __init__(self, input_size=python_lang.n_words, d_model=512, ffn_hidden=2048, num_heads=8, drop_prob=0.1, num_layers=5):
      super().__init__()
      self.embedding = nn.Embedding(input_size, d_model)
      self.positionalembedding = nn.Embedding(64, d_model)
      self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                        for _ in range(num_layers)])
      self.unembedding = nn.Linear(d_model, python_lang.n_words)
      self.d_model = d_model

  def forward(self, x, y, mask, mask2):
      y = self.embedding(y)
      batch_size = len(y)
      # positional = PositionalEncoding(self.d_model, 51)
      # y = y + positional().unsqueeze(0).repeat(batch_size, 1, 1)
      positional = torch.arange(64).unsqueeze(0)
      positional = self.positionalembedding(positional)
      positional = positional.repeat(batch_size, 1, 1)
      y = y + positional
      y = self.layers(x, y, mask, mask2)
      y = self.unembedding(y)
      return y #30 x 200 x 6873