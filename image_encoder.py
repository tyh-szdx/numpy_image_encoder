import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector

class MultiHeadAttention:
    def __init__(self, weights, prefix, heads):
        self.qkv_weight = weights[prefix + '.in_proj_weight']  # [3D,D]
        self.qkv_bias = weights[prefix + '.in_proj_bias']      # [3D]
        self.out_weight = weights[prefix + '.out_proj.weight']  # [D,D]
        self.out_bias = weights[prefix + '.out_proj.bias']
        self.heads = heads
        self.width = self.out_bias.shape[0]
        self.head_dim = self.width // heads

    def split_heads(self, x):
        batch, tokens, width = x.shape
        return x.reshape(batch, tokens, self.heads, self.head_dim).transpose(0, 2, 1, 3)

    def __call__(self, x, causal=False):
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2*self.width])
        v = self.split_heads(qkv[:, :, 2*self.width:])

        scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(np.float32(self.head_dim))

        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)  

        context = coefficients @ v
        merged = context.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], self.width)

        return merged @ self.out_weight.T + self.out_bias


class TransformerBlock:
    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix+'.ln_1.weight'], weights[prefix+'.ln_1.bias'])
        self.norm2 = LayerNorm(weights[prefix+'.ln_2.weight'], weights[prefix+'.ln_2.bias'])
        self.attention = MultiHeadAttention(weights, prefix+'.attn', heads)
        self.fc1_weight = weights[prefix+'.mlp.c_fc.weight']  # [4D,D]
        self.fc1_bias = weights[prefix+'.mlp.c_fc.bias']
        self.fc2_weight = weights[prefix+'.mlp.c_proj.weight']  # [D,4D]
        self.fc2_bias = weights[prefix+'.mlp.c_proj.bias']
        self.gelu = GELU()

    def __call__(self, x, causal=False):
        update = self.attention(self.norm1(x), causal=causal)

        residual = x + update

        u = self.norm2(residual)

        hidden = self.gelu(u @ self.fc1_weight.T + self.fc1_bias)
        mlp_update = hidden @ self.fc2_weight.T + self.fc2_bias
        return residual + mlp_update


class ImageEncoder:
    def __init__(self, weights):
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072)
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768)
        self.position = weights['visual.positional_embedding']  # [50,768]
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj']  

    def patchify(self, images):
        batch = images.shape[0]
        patches = images.reshape(batch, 3, 7, 32, 7, 32)
        patches = patches.transpose(0, 2, 4, 1, 3, 5)
        return patches.reshape(batch, 49, 3072)

    def embed(self, images):
        patches = self.patchify(images)  # [B,49,3072]
        patch_tokens = patches @ self.patch_weight.T

        cls = np.tile(self.cls, (images.shape[0], 1, 1))

        tokens = np.concatenate((cls, patch_tokens), axis=1)
        return tokens + self.position

    def encode_tokens(self, images):
        x = self.pre_norm(self.embed(images))
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):
        tokens = self.encode_tokens(images)

        cls_token = tokens[:, 0, :]
        normalized = self.post_norm(cls_token[:, None, :])[:, 0, :]
        output = normalized @ self.projection

        return unit_vector(output)
