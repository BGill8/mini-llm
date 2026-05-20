import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, init_len=1024, base=10000):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        inv_freq = 1.0 / (base ** (2*torch.arange(0, head_dim//2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.build_cache(init_len)

    def build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None], persistent=False)
        # can reference these as self.cos_cached and self.sin_cached
        # just don't want to save them with the model so made this way

    

    def forward(self, q, k):
        Tseq = q.shape[2]
        
        # Rebuild cache if sequence is longer than cached
        if Tseq > self.cos_cached.shape[2]:
            self.build_cache(Tseq)
        
        cos = self.cos_cached[:, :, :Tseq, :]  # 1 x 1 x T x d
        sin = self.sin_cached[:, :, :Tseq, :]
        
        # Helper: rotate_half splits the last dim in half and returns [-x2, x1]
        def rotate_half(x):
            half = x.shape[-1] // 2
            x1, x2 = x[..., :half], x[..., half:]
            return torch.cat([-x2, x1], dim=-1)
        
        rot_q = q * cos + rotate_half(q) * sin
        rot_k = k * cos + rotate_half(k) * sin
        
        return rot_q, rot_k



###########
# Q2
###########
class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, rope, n_groups):
        super().__init__()
        self.n_heads  = n_heads
        self.n_groups = n_groups
        self.head_dim = dim // n_heads
        
        self.qkv = nn.Linear(dim, self.head_dim * (n_heads + 2 * n_groups), bias=False)
        self.out  = nn.Linear(dim, dim, bias=False)
        self.rope = rope
        self.qnorm = nn.RMSNorm(self.head_dim)
        self.knorm = nn.RMSNorm(self.head_dim)
        
        
    def forward(self, x):
        B, T, C = x.shape
        
        # 1. Fused projection and split
        z = self.qkv(x)  # B x T x head_dim*(H + 2G)
        q, k, v = torch.split(
            z, 
            [self.head_dim * self.n_heads,
            self.head_dim * self.n_groups,
            self.head_dim * self.n_groups],
            dim=-1
        )
        
        # Reshape to (B, heads, T, head_dim)
        q = q.view(B, T, self.n_heads,  self.head_dim).transpose(1, 2)  # B x H x T x d
        k = k.view(B, T, self.n_groups, self.head_dim).transpose(1, 2)  # B x G x T x d
        v = v.view(B, T, self.n_groups, self.head_dim).transpose(1, 2)  # B x G x T x d
        
        # 2. RoPE then QK norm
        q, k = self.rope(q, k)
        q = self.qnorm(q)
        k = self.knorm(k)
        
        # 3. Flash attention with GQA
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        # a: B x H x T x head_dim
        
        # 4. Merge heads and project
        a = a.transpose(1, 2).contiguous().view(B, T, C)  # B x T x dim
        out = self.out(a)
        
        return out.contiguous()


###########
# Q3
###########
class FeedForwardBlock(nn.Module):
    def __init__(self, dim, bottleneck_factor=2):
        super().__init__()
        self.w12 = nn.Linear(dim, 2 * dim * bottleneck_factor, bias=False)
        self.out  = nn.Linear(dim * bottleneck_factor, dim, bias=False)

    def forward(self, x):
        z1, z2 = torch.chunk(self.w12(x), 2, dim=-1)  # each B x T x (dim*factor)
        return self.out(z1 * F.silu(z2))


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, rope, bottleneck_factor=2, n_groups=2):
        super().__init__()
        self.ln1  = nn.RMSNorm(dim)
        self.ln2  = nn.RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads, rope, n_groups)
        self.ffn  = FeedForwardBlock(dim, bottleneck_factor)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, dim=704, depth=12, n_heads=8, max_len=1024):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        rope = RotaryEmbedding(dim // n_heads, init_len=max_len)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, n_heads, rope) for _ in range(depth)]
        )
        self.ln_f = nn.RMSNorm(dim)
        
        self.init_weights()
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)

    def init_weights(self):
        for name, module in self.named_modules():
            if "out" in name:
                # Depth scaling 
                nn.init.normal_(module.weight, mean=0.0, std=0.02/math.sqrt(len(self.blocks)))
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.RMSNorm):
                nn.init.ones_(module.weight)

    def forward(self, idx):
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return F.linear(x, self.tok_emb.weight)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temp=1.0, top_p=0.9, max_context=1024):
        for _ in range(max_new_tokens):
            logits = self(idx[:, -max_context:])[:, -1, :] / temp
            probs  = torch.softmax(logits, dim=-1)

            ###########
            # Q4
            ###########
            # Sort probabilities descending
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)

            # Cumulative sum; find cutoff where cumsum first >= top_p
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            # Shift right so the token that pushes us over the threshold is kept
            cutoff_mask = (cumsum - sorted_probs) >= top_p  # True = remove

            # Zero out tokens outside the nucleus
            sorted_probs[cutoff_mask] = 0.0

            # Scatter back to original vocabulary ordering
            probs_filtered = torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_probs)

            # Renormalize and sample
            next_token = torch.multinomial(probs_filtered, num_samples=1)  # B x 1

            idx = torch.cat([idx, next_token], dim=1)
        return idx


if __name__=="__main__":

    # Quick unit test to see if RoPE actually only cares about relative position
    # Also checks if the cache automatically expands
    dim = 128
    rope = RotaryEmbedding(head_dim=dim, init_len=64)
 
    q_vec = torch.randn(1, 1, 1, dim)
    k_vec = torch.randn(1, 1, 1, dim)

    def rotated_dot(q_pos, k_pos):
        q_seq = torch.zeros(1, 1, q_pos + 1, dim)
        k_seq = torch.zeros(1, 1, k_pos + 1, dim)
        q_seq[..., q_pos, :] = q_vec
        k_seq[..., k_pos, :] = k_vec
        q_r, _ = rope(q_seq, q_seq)
        _, k_r = rope(k_seq, k_seq)
        return (q_r[..., q_pos, :] * k_r[..., k_pos, :]).sum()

    dot_0_3 = rotated_dot(0, 3)   # relative offset = 3
    dot_2_5 = rotated_dot(52, 55)   # relative offset = 3
    assert torch.isclose(dot_0_3, dot_2_5, atol=1e-5), (
        f"RoPE inner products differ for same relative offset: "
        f"{dot_0_3.item():.6f} vs {dot_2_5.item():.6f}"
    )