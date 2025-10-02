import torch
from torch import nn

class AttentionHead(nn.Module):
	def __init__(self, input_dim, head_dim):
		super(AttentionHead, self).__init__()
		self.query = nn.Linear(input_dim, head_dim)
		self.key = nn.Linear(input_dim, head_dim)
		self.value = nn.Linear(input_dim, head_dim)
		self.scale = head_dim ** -0.5

	def forward(self, x):
		Q = self.query(x)
		K = self.key(x)
		V = self.value(x)

		attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
		attn_weights = torch.softmax(attn_weights, dim=-1)

		out = torch.matmul(attn_weights, V)
		return out
	
class MultiHeadAttention(nn.Module):
	def __init__(self, input_dim, head_dim, num_heads):
		super(MultiHeadAttention, self).__init__()
		self.heads = nn.ModuleList([AttentionHead(input_dim, head_dim) for _ in range(num_heads)])
		self.linear = nn.Linear(head_dim * num_heads, input_dim)

	def forward(self, x):
		head_outputs = [head(x) for head in self.heads]
		concat = torch.cat(head_outputs, dim=-1)
		out = self.linear(concat)
		return out
	
class FeedForward(nn.Module):
	def __init__(self, input_dim, ff_dim):
		super(FeedForward, self).__init__()
		self.fc1 = nn.Linear(input_dim, ff_dim)
		self.fc2 = nn.Linear(ff_dim, input_dim)
		self.activation = nn.ReLU()

	def forward(self, x):
		out = self.fc1(x)
		out = self.activation(out)
		out = self.fc2(out)
		return out
	
class TransformerBlock(nn.Module):
	def __init__(self, input_dim, head_dim, num_heads, ff_dim, dropout=0.1):
		super(TransformerBlock, self).__init__()
		self.mha = MultiHeadAttention(input_dim, head_dim, num_heads)
		self.ffn = FeedForward(input_dim, ff_dim)
		self.norm1 = nn.LayerNorm(input_dim)
		self.norm2 = nn.LayerNorm(input_dim)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		attn_out = self.mha(x)
		x = self.norm1(x + self.dropout(attn_out))
		ffn_out = self.ffn(x)
		x = self.norm2(x + self.dropout(ffn_out))
		return x
	
class Transformer(nn.Module):
	def __init__(self, input_dim, head_dim, num_heads, ff_dim, depth, dropout=0.1):
		super(Transformer, self).__init__()
		self.layers = nn.ModuleList([
			TransformerBlock(input_dim, head_dim, num_heads, ff_dim, dropout) for _ in range(depth)
		])

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
		return x