"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import jax
import jax.numpy as jnp

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))  # weight: (normalized_dim,)
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None  # bias: (normalized_dim,)

    def forward(self, input):
        # input: (batch_size, seq_len, normalized_dim)
        mean = jnp.mean(input, axis=-1, keepdims=True)  # (batch_size, seq_len, 1)
        var = jnp.mean((input - mean) ** 2, axis=-1, keepdims=True)  # (batch_size, seq_len, 1)
        inv_std = jax.lax.rsqrt(var + 1e-5)  # (batch_size, seq_len, 1)
        x_norm = (input - mean) * inv_std  # (batch_size, seq_len, normalized_dim)
        y = x_norm * jnp.asarray(self.weight)  # (batch_size, seq_len, normalized_dim)
        if self.bias is not None:
            y = y + jnp.asarray(self.bias)  # (batch_size, seq_len, normalized_dim)
        return y  # (batch_size, seq_len, normalized_dim)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.flash = False
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # (batch_size, seq_len, embed_dim)
        qkv = self.c_attn(x)  # (batch_size, seq_len, 3*embed_dim)
        q, k, v = jnp.split(qkv, 3, axis=2)  # (batch_size, seq_len, embed_dim) each

        head_dim = C // self.n_head  # head_dim = embed_dim / num_heads

        k = jnp.transpose(jnp.reshape(k, (B, T, self.n_head, head_dim)), (0, 2, 1, 3))  # (batch_size, num_heads, seq_len, head_dim)
        q = jnp.transpose(jnp.reshape(q, (B, T, self.n_head, head_dim)), (0, 2, 1, 3))  # (batch_size, num_heads, seq_len, head_dim)
        v = jnp.transpose(jnp.reshape(v, (B, T, self.n_head, head_dim)), (0, 2, 1, 3))  # (batch_size, num_heads, seq_len, head_dim)

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)  # (batch_size, num_heads, seq_len, head_dim)
        else:
            # manual implementation of attention
            att = jnp.matmul(q, jnp.swapaxes(k, -2, -1))  # (batch_size, num_heads, seq_len, seq_len): dot-product attention scores
            att = att * (1.0 / jnp.sqrt(head_dim))  # (batch_size, num_heads, seq_len, seq_len): scale by sqrt(head_dim)

            causal_bias = self.bias[:, :, :T, :T]  # (1, 1, seq_len, seq_len)
            att = jnp.where(causal_bias == 0, jnp.full_like(att, -jnp.inf), att)  # (batch_size, num_heads, seq_len, seq_len): apply causal mask

            att = jax.nn.softmax(att, axis=-1)  # (batch_size, num_heads, seq_len, seq_len): normalize over key positions
            # TODO: JAX dropout requires an explicit PRNG key; using PyTorch dropout module call as a placeholder.
            att = self.attn_dropout(att)  # (batch_size, num_heads, seq_len, seq_len)

            y = jnp.matmul(att, v)  # (batch_size, num_heads, seq_len, head_dim): weighted sum of values

        y = jnp.transpose(y, (0, 2, 1, 3))  # (batch_size, seq_len, num_heads, head_dim)
        y = jnp.reshape(y, (B, T, C))  # (batch_size, seq_len, embed_dim)

        # output projection
        y = self.c_proj(y)  # (batch_size, seq_len, embed_dim)
        # TODO: JAX dropout requires an explicit PRNG key; using PyTorch dropout module call as a placeholder.
        y = self.resid_dropout(y)  # (batch_size, seq_len, embed_dim)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)  # (batch_size, seq_len, n_embd) -> (batch_size, seq_len, hidden_dim_ffn)
        x = jax.nn.gelu(x)  # (batch_size, seq_len, hidden_dim_ffn) -> (batch_size, seq_len, hidden_dim_ffn)
        x = self.c_proj(x)  # (batch_size, seq_len, hidden_dim_ffn) -> (batch_size, seq_len, n_embd)
        x = self.dropout(x)  # TODO: JAX dropout requires an explicit PRNG key; (batch_size, seq_len, n_embd) -> (batch_size, seq_len, n_embd)
        return x  # (batch_size, seq_len, n_embd)

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        x_norm_1 = self.ln_1(x)  # (batch_size, seq_len, embed_dim)
        attn_out = self.attn(x_norm_1)  # (batch_size, seq_len, embed_dim)
        x = x + attn_out  # (batch_size, seq_len, embed_dim)

        x_norm_2 = self.ln_2(x)  # (batch_size, seq_len, embed_dim)
        mlp_out = self.mlp(x_norm_2)  # (batch_size, seq_len, embed_dim)
        x = x + mlp_out  # (batch_size, seq_len, embed_dim)

        return x  # (batch_size, seq_len, embed_dim)

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx: (batch_size, seq_len) token indices
        batch_size, seq_len = idx.shape  # (batch_size, seq_len)
        assert seq_len <= self.config.block_size, f"Cannot forward sequence of length {seq_len}, block size is only {self.config.block_size}"

        # pos: (seq_len) position indices
        pos = jnp.arange(0, seq_len, dtype=jnp.int32)  # (seq_len)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # (batch_size, seq_len, hidden_dim)
        pos_emb = self.transformer.wpe(pos)  # (seq_len, hidden_dim)
        x = tok_emb + pos_emb  # (batch_size, seq_len, hidden_dim)
        x = self.transformer.drop(x)  # (batch_size, seq_len, hidden_dim)  # TODO: JAX dropout needs an explicit PRNG key

        for block in self.transformer.h:
            x = block(x)  # (batch_size, seq_len, hidden_dim)

        x = self.transformer.ln_f(x)  # (batch_size, seq_len, hidden_dim)

        if targets is not None:
            # logits: (batch_size, seq_len, vocab_size)
            logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)

            # flatten for cross-entropy: (batch_size*seq_len, vocab_size) and (batch_size*seq_len,)
            logits_flat = logits.reshape(-1, logits.shape[-1])  # (batch_size*seq_len, vocab_size)
            targets_flat = targets.reshape(-1)  # (batch_size*seq_len,)

            # cross entropy with ignore_index=-1
            log_probs = jax.nn.log_softmax(logits_flat, axis=-1)  # (batch_size*seq_len, vocab_size)
            valid_mask = (targets_flat != -1)  # (batch_size*seq_len,)
            safe_targets = jnp.where(valid_mask, targets_flat, 0)  # (batch_size*seq_len,)
            nll = -jnp.take_along_axis(log_probs, safe_targets[:, None], axis=-1).squeeze(-1)  # (batch_size*seq_len,)
            nll = jnp.where(valid_mask, nll, 0.0)  # (batch_size*seq_len,)
            loss = jnp.sum(nll) / jnp.maximum(jnp.sum(valid_mask), 1)  # ()

        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            x_last = x[:, [-1], :]  # (batch_size, 1, hidden_dim)
            logits = self.lm_head(x_last)  # (batch_size, 1, vocab_size)
            loss = None  # None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        config_args['bias'] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        # NOTE: This method uses PyTorch sampling utilities; JAX sampling would require an explicit PRNG key.
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature  # (batch_size, vocab_size)
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)  # (batch_size, vocab_size)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, seq_len + 1)

        return idx
