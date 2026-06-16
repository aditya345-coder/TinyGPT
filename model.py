import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


config = {
    "vocab_size": 50257,
    "block_size": 512,
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
}


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, (self.weight.shape[0],), self.weight, self.bias, eps=1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config["n_head"]
        self.n_embd = config["n_embd"]

        self.c_attn = nn.Linear(config["n_embd"], 3 * config["n_embd"])
        self.c_proj = nn.Linear(config["n_embd"], config["n_embd"])

        mask = torch.tril(torch.ones(config["block_size"], config["block_size"]))
        self.register_buffer("bias", mask.view(1, 1, config["block_size"], config["block_size"]))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config["n_embd"], 4 * config["n_embd"])
        self.c_proj = nn.Linear(4 * config["n_embd"], config["n_embd"])

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config["n_embd"])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config["n_embd"])
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config["vocab_size"], config["n_embd"])
        self.wpe = nn.Embedding(config["block_size"], config["n_embd"])
        self.h = nn.ModuleList([Block(config) for _ in range(config["n_layer"])])
        self.ln_f = LayerNorm(config["n_embd"])
        self.lm_head = nn.Linear(config["n_embd"], config["vocab_size"], bias=False)

        self.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb

        for block in self.h:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config["block_size"]:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def load_pretrained_weights(model):
    print("Loading pretrained GPT-2 weights...")
    hf_model = AutoModelForCausalLM.from_pretrained("gpt2")
    hf_sd = hf_model.state_dict()

    our_sd = model.state_dict()
    copied = 0
    for hf_name, hf_param in hf_sd.items():
        our_name = hf_name.replace("transformer.", "")
        if our_name in our_sd and hf_param.shape == our_sd[our_name].shape:
            our_sd[our_name] = hf_param
            copied += 1

    model.load_state_dict(our_sd, strict=False)
    print(f"Copied {copied} weight tensors from pretrained GPT-2")
    return model
