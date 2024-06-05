import torch
from torch import nn
import torch.nn.functional as F
import math

#----------------------------------------------------------
# Activation
#----------------------------------------------------------
def apply_activation(x, activation='linear'):
    if activation == 'linear':
        return x
    elif activation == 'gelu_new':
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    elif activation == 'gelu_fast':
        return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))
    elif activation == 'gelu':
        return F.gelu(x)
    elif activation == 'relu':
        return F.relu(x)
    elif activation == 'leaky_relu':
        return F.leaky_relu(x)
    elif activation == 'sigmoid':
        return F.sigmoid(x)
    elif activation == 'tanh':
        return F.tanh(x)
    elif activation == 'softmax':
        return F.softmax(x, dim=-1)
    else:
        raise ValueError(f'Unknown activation function: {activation}.')

#----------------------------------------------------------
# Attention
#----------------------------------------------------------
def split_heads(x, num_heads, head_dim):
    """
    Splits embeddings for different heads.

    Args:
        x (tensor): Input tensor, shape [B, seq_len, embd_dim] or [B, blocks, block_len, embd_dim].
        num_heads (int): Number of heads.
        head_dim (int): Dimension of embedding for each head.

    Returns:
        (tensor): Output tensor, shape [B, num_head, seq_len, head_dim] or [B, blocks, num_head, block_len, head_dim].
    """
    newshape = x.shape[:-1] + (num_heads, head_dim)
    x = x.view(newshape)
    if x.ndim == 5:
        # [batch, blocks, head, block_len, head_dim]
        return x.permute(0, 1, 3, 2, 4)
    elif x.ndim == 4:
        # [batch, head, seq_len, head_dim]
        return x.permute(0, 2, 1, 3)
    else:
        raise ValueError(f'Input tensor should have rank 4 or 5, but has rank {x.ndim}.')


def merge_heads(x, num_heads, head_dim):
    """
    Merge embeddings for different heads.

    Args:
        x (tensor): Input tensor, shape [B, num_head, seq_len, head_dim] or [B, blocks, num_head, block_len, head_dim].
        num_heads (int): Number of heads.
        head_dim (int): Dimension of embedding for each head.

    Returns:
        (tensor): Output tensor, shape [B, seq_len, embd_dim] or [B, blocks, block_len, embd_dim].
    """
    if x.ndim == 5:
        x = torch.permute(x, dims=(0, 1, 3, 2, 4))
    elif x.ndim == 4:
        x = torch.permute(x, dims=(0, 2, 1, 3))
    else:
        raise ValueError(f'Input tensor should have rank 4 or 5, but has rank {x.ndim}.')

    newshape = x.shape[:-2] + (num_heads * head_dim,)
    x = torch.reshape(x, newshape)
    return x


def attention(query, key, value, casual_mask, masked_bias, dropout, scale_attn_weights, attn_mask=None, head_mask=None, feedback=None):
    """
    Computes Dot-Product Attention for the given query, key and value.
    
    Args:
        query (tensor): Query, shape [B, num_heads, seq_len, embd_dim].
        key (tensor): Key, shape [B, num_heads, seq_len, embd_dim].
        value (tensor): Value, shape [B, num_heads, seq_len, embd_dim].
        casual_mask (tensor): Mask to ensure that attention is only applied to the left of the input sequence, 
                              shape [1, 1, key_len - query_len :key_len, :key_len].
        masked_bias (float): Value to insert for masked part of the sequence.
        dropout (nn.Dropout): Dropout module that is applied to the attention output.
        scale_attn_weights (bool): If True, scale the attention weights.
        training (bool): Training mode.
        attn_mask (tensor): Mask to avoid performing attention on padded tokens indices, shape [B, seq_len].
        head_mask (tensor): Mask to nullify selected heads of the self-attention modules, shape [num_heads,] or [num_layers, num_heads].
        feedback (tensor): external feedback with marked points.

    Returns:
        (tensor): Attention output, shape [B, num_heads, seq_len, embd_dim].
        (tensor): Attention weights, shape [B, num_heads, seq_len, seq_len].
        (tensor): KLD loss with external feedback, float.
    """
    masked_bias = torch.tensor(masked_bias, dtype=torch.float32, device=query.device)
    query = query.to(dtype=torch.float32)
    key = key.to(dtype=torch.float32)
    attn_weights = torch.matmul(query, torch.swapaxes(key, -1, -2))

    if scale_attn_weights:
        attn_weights = attn_weights / (float(value.shape[-1]) ** 0.5)

    casual_mask = casual_mask.to(device=attn_weights.device)
    attn_weights = torch.where(casual_mask, attn_weights, masked_bias)

    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask.cuda()
        # attn_weights = attn_weights + attn_mask
   
    _attn_weights = F.softmax(attn_weights, dim=-1)
    attn_weights = _attn_weights.type(value.dtype)
    attn_weights = dropout(attn_weights)

    if head_mask is not None:
        attn_weights = attn_weights * head_mask.cuda()
        # attn_weights = attn_weights * head_mask

    out = torch.matmul(attn_weights, value)
    return out, _attn_weights 

#----------------------------------------------------------
# Misc
#----------------------------------------------------------

def get_attention_mask(attn_mask, batch_size):
    assert batch_size > 0, 'batch_size should be > 0.'
    attn_mask = torch.reshape(attn_mask, shape=(batch_size, -1))
    attn_mask = attn_mask[:, None, None, :]
    attn_mask = (1.0 - attn_mask) * -10000.0
    return attn_mask


def get_head_mask(head_mask, num_layers):
    if head_mask.ndim == 1:
        head_mask = head_mask[None, None, :, None, None]
        head_mask = torch.repeat_interleave(head_mask, num_layers, dim=0)
    elif head_mask.ndim == 2:
        head_mask = head_mask[:, None, :, None, None]
    else:
        raise ValueError(f'head_mask must have rank 5, but has rank {head_mask.ndim}.')
    return head_mask

