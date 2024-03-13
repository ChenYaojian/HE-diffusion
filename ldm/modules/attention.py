from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from ldm.modules.diffusionmodules.util import checkpoint
from ldm.enc_util import enc_sum


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        #print("x shape: ", x.shape)
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        #print("q shape: ", q.shape)
        #print("k shape: ", k.shape)
        #print("v shape: ", v.shape)
        bq, nq, hdq = q.shape
        dq = hdq // h
        bk, nk, hdk = k.shape
        dk = hdk // h
        bv, nv, hdv = v.shape
        dv = hdv // h

        #q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        q = q.reshape(bq, nq*h, dq).permute(2,1,0).reshape(dq,nq,h,bq).permute(3,2,1,0).reshape(bq*h, nq,dq)
        k = k.reshape(bk, nk*h, dk).permute(2,1,0).reshape(dk,nk,h,bk).permute(3,2,1,0).reshape(bk*h, nk,dk)
        v = v.reshape(bv, nv*h, dv).permute(2,1,0).reshape(dv,nv,h,bv).permute(3,2,1,0).reshape(bv*h, nv,dv)

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = mask.reshape(mask.shape[0], -1)
            #mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class EncryptedCrossAttention(nn.Module):
    def __init__(self, torch_nn, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q_weight = torch_nn.to_q.weight.T.data.tolist()
        self.to_q_bias = torch_nn.to_q.bias.data.tolist()
        self.to_k_weight = torch_nn.to_k.weight.T.data.tolist()
        self.to_k_bias = torch_nn.to_k.bias.data.tolist()
        self.to_v_weight = torch_nn.to_v.weight.T.data.tolist()
        self.to_v_bias = torch_nn.to_v.bias.data.tolist()
        self.to_out_weight = torch_nn.to_out[0].weight.T.data.tolist()
        self.to_out_bias = torch_nn.to_out[0].bias.data.tolist()

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = x.mm(self.to_q_weight) + self.to_q_bias
        context = default(context, x)
        k = context.mm(self.to_k_weight) + self.to_k_bias
        v = context.mm(self.to_v_weight) + self.to_v_bias
        bq, nq, hdq = q.shape
        dq = hdq // h
        bk, nk, hdk = k.shape
        dk = hdk // h
        bv, nv, hdv = v.shape
        dv = hdv // h

        #q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        q = q.reshape(bq, nq*h, dq).transpose().reshape(dq,nq,h,bq).transpose().reshape(bq*h, nq,dq)
        k = k.reshape(bk, nk*h, dk).transpose().reshape(dk,nk,h,bk).transpose().reshape(bk*h, nk,dk)
        v = v.reshape(bv, nv*h, dv).transpose().reshape(dv,nv,h,bv).transpose().reshape(bv*h, nv,dv)

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        out = out.mm(self.to_out_weight) + self.to_out_bias
        return out

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x

class EncryptedBasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        # Encrypted components
        self.enc_attn1 = EncryptedAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # Simplified
        self.enc_ff = EncryptedFeedForward(dim, dropout=dropout, glu=gated_ff)  # Simplified, linear only
        self.enc_attn2 = EncryptedAttention(query_dim=dim, context_dim=context_dim,
                                            heads=n_heads, dim_head=d_head, dropout=dropout)  # Simplified
        self.enc_norm1 = EncryptedLayerNorm(dim)  # Highly simplified
        self.enc_norm2 = EncryptedLayerNorm(dim)  # Highly simplified
        self.enc_norm3 = EncryptedLayerNorm(dim)  # Highly simplified
        self.checkpoint = checkpoint

    def forward(self, encrypted_input, context=None):
        x = self.enc_attn1(self.enc_norm1(encrypted_input)) + encrypted_input
        x = self.enc_attn2(self.enc_norm2(x), context=context) + x
        x = self.enc_ff(self.enc_norm3(x)) + x
        return x

def enc_bmm(enc_a, enc_b):
    #do bmm as einsum('b i d, b j d -> b i j', q, k)
    assert len(enc_a.shape) == len(enc_b.shape)
    assert len(enc_a.shape) == 3
    b,n,d = enc_a.shape
    result = []
    for i in range(b):
        enc_a_i = enc_a[i]
        enc_b_i = enc_b[i]
        enc_c = enc_a_i.mm(enc_b_i)
        result.append(enc_c)
    return result
    
class EncryptedLayerNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-5, gamma=1, beta=0):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.gamma = gamma
        self.beta = beta

    def forward(self, encrypted_input):
        # Approximate mean
        sum_vector = enc_sum(encrypted_input)
        mean = sum_vector * (1 / self.dim)

        # Approximate variance
        diff = encrypted_input - mean
        sq_diff = diff * diff
        sum_sq_diff = enc_sum(sq_diff)
        variance = sum_sq_diff * (1 / self.dim)

        # Adding epsilon for numerical stability
        variance_eps = variance + self.eps

        # Approximate inverse square root using a polynomial
        # Choosing the right polynomial coefficients is crucial for accuracy
        inv_sqrt_var = 1 / math.sqrt(variance_eps.decrypt().tolist())

        # Normalizing
        normalized = (encrypted_input - mean) * inv_sqrt_var

        # Apply scale and shift
        result = self.gamma * normalized + self.beta
        return result


class EncryptedFeedForward(torch.nn.Module):
    def __init__(self, feed_net):
        super().__init__()
        self.weight1 = feed_net.net[0][0].weight.T.data.tolist()
        self.bias1 = feed_net.net[0][0].bias.T.data.tolist()
        self.weight2 = feed_net.net[2].weight.T.data.tolist()
        self.bias2 = feed_net.net[2].bias.T.data.tolist()

    def forward(self, encrypted_input):
        # First linear layer
        encrypted_output = encrypted_input.mm(self.weight1) + self.bias1

        # Apply degree-3 polynomial approximation for ReLU
        encrypted_output = self.poly_relu(encrypted_output)

        # Second linear layer
        encrypted_output = encrypted_output.mm(self.weight2) + self.bias2

        return encrypted_output

    def poly_relu(self, encrypted_input):
        # Degree-3 polynomial coefficients for ReLU approximation
        relu_coeffs = [0, 0.5, 0.5, -0.125]
        return ts.polyval(encrypted_input, relu_coeffs)


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in
