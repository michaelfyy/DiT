import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.vision_transformer import PatchEmbed
from einops import rearrange


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class Time_Embedder(nn.Module):
    def __init__(
            self,
            embedding_dim=768,
            frequency=256,
    ):
        super().__init__()
        self.frequency = frequency
        self.embedding_dim = embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def timestep_embedding(self, t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, x):
        x = self.timestep_embedding(x, self.frequency)
        x = self.mlp(x)
        return x # x = (B, D)


class Label_Embedder(nn.Module):
    def __init__(
            self,
            num_classes=1000,
            embedding_dim=768,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.num_clases = num_classes

    def forward(self, x):
        return self.embedding(x) # x = (B, D)


class AdaLN(nn.Module):
    def __init__(self, embedding_dim=768):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_dim, 6*embedding_dim)
        )

    def forward(self, x): # x = (B, D)
        x = self.mlp(x) # (B, 6*D)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = x.unsqueeze(1).chunk(6, dim=2)
        return gamma1, beta1, alpha1, gamma2, beta2, alpha2 # each is (B, 1, D)


class DiT_Block_AdaLN(nn.Module): # like in-context but no conditioning
    def __init__(
        self,
        embedding_dim=768,
        num_heads=12
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        self.mha = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.adaptive_ln = AdaLN(embedding_dim=embedding_dim)

        self.pointwise_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(approximate='tanh'),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
          
    def forward(self, x, c):

        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaptive_ln(c) # each one is (B, D)
        residual = x

        x = self.norm1(x) # layer norm 1
        x = (1+gamma1)*x + beta1 # scale & shift 1
        attn_output, attn_weights = self.mha(x, x, x)
        x = residual + alpha1*attn_output

        residual = x
        x = self.norm2(x) # layer norm 2
        x = (1+gamma2)*x + beta2 # scale & shift 2
        x = residual + alpha2*self.pointwise_mlp(x)

        return x
 

class FinalLayer(nn.Module):
    def __init__(
            self,
            embedding_dim=768,
            patch_size=4,
            img_size=32,
            num_channels=4
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_channels = num_channels
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        self.adpative_ln_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_dim, 2*embedding_dim)
        )
        self.linear = nn.Linear(embedding_dim, patch_size*patch_size*2*num_channels)

    def forward(self, x, c):
        x = self.norm(x)
        gamma, beta = self.adpative_ln_mlp(c).unsqueeze(1).chunk(2, dim=2)
        x = (1+gamma)*x + beta
        x = self.linear(x)
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
                      p1=self.patch_size, p2=self.patch_size, h=self.img_size // self.patch_size, w=self.img_size // self.patch_size)
        return x
    

class DiT(nn.Module):
    def __init__(
            self,
            embedding_dim=768,
            num_heads=12,
            patch_size=4,
            n_blocks=28,
            num_classes=1000,
            frequency=256,
            img_size=32,
            in_chans=4,
    ):
        super().__init__()  

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.img_size = img_size
        self.in_chans = in_chans

        self.patch_embedder = PatchEmbed(img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=embedding_dim)
        self.positional_embedding = get_2d_sincos_pos_embed(embedding_dim, img_size//patch_size)
        self.positional_embedding = torch.from_numpy(self.positional_embedding).float().unsqueeze(0)
        self.time_embedder = Time_Embedder(embedding_dim=embedding_dim, frequency=frequency)
        self.label_embedder = Label_Embedder(num_classes=num_classes, embedding_dim=embedding_dim)
        self.dit_blocks = nn.ModuleList([
            DiT_Block_AdaLN(embedding_dim=embedding_dim, num_heads=num_heads) for _ in range(n_blocks)
        ])
        self.final_layer = FinalLayer(embedding_dim=embedding_dim, patch_size=patch_size, img_size=img_size, num_channels=in_chans)

        # initialize weights
        self.init_weights()

    def init_weights(self):
        # GLobal Xavier init for all Linear layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.patch_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patch_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.label_embedder.embedding.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)

        # Zero-out AdaLN weights and biases
        for block in self.dit_blocks:
            nn.init.constant_(block.adaptive_ln.mlp[-1].weight, 0)
            nn.init.constant_(block.adaptive_ln.mlp[-1].bias, 0)

        # Zero-out FinalLayer
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        nn.init.constant_(self.final_layer.adpative_ln_mlp[-1].weight, 0)
        nn.init.constant_(self.final_layer.adpative_ln_mlp[-1].bias, 0)

    def embed_x(self, x):
        x = self.patch_embedder(x)  # (B, T, D)
        x += self.positional_embedding.to(x.device)
        return x

    def forward(self, x, time, label): # x.shape = torch.Size[(B, 4, 32, 32)] (B, C, H, W)
        x = self.embed_x(x)
        time_embedding = self.time_embedder(time)
        label_embedding = self.label_embedder(label)

        conditioning = time_embedding + label_embedding

        for dit_block in self.dit_blocks:
            x = dit_block(x, conditioning)

        x = self.final_layer(x, conditioning)
        return x
    

#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(n_blocks=28, embedding_dim=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(n_blocks=28, embedding_dim=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(n_blocks=28, embedding_dim=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(n_blocks=24, embedding_dim=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(n_blocks=24, embedding_dim=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(n_blocks=24, embedding_dim=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(n_blocks=12, embedding_dim=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(n_blocks=12, embedding_dim=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(n_blocks=12, embedding_dim=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(n_blocks=12, embedding_dim=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(n_blocks=12, embedding_dim=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(n_blocks=12, embedding_dim=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}
    
    