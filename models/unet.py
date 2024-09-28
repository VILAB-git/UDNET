import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm

from functools import wraps
from packaging import version
from collections import namedtuple

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D


ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
AttentionConfig = namedtuple('AttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


class Upsample(nn.Module):
    """
    ### Scale up the feature map by $2 \times$
    """

    def __init__(self, dim, dim_out = None):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, default(dim_out, dim), kernel_size=4, stride=2, padding=1)
        self.act = nn.ReLU()
        self.norm = nn.InstanceNorm2d(default(dim_out, dim))

    def forward(self, x):
        return self.norm(self.conv(self.act(x)))


class Downsample(nn.Module):
    """
    ### Scale down the feature map by $\frac{1}{2} \times$
    """

    def __init__(self, dim, dim_out = None):
        super().__init__()
        # self.conv = nn.Conv2d(dim, default(dim_out, dim), (3, 3), (2, 2), (1, 1))
        self.conv = nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)
        self.act = nn.ReLU()
        self.norm = nn.InstanceNorm2d(default(dim_out, dim))

    def forward(self, x):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        # _ = t
        # _ = z
        return self.norm(self.conv(self.act(x)))

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta


    def forward(self, x, theta=None):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8, dropout=0.0, temp_nc=3):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1, groups=temp_nc)
        self.norm = nn.InstanceNorm2d(dim_out)
        self.drop = nn.Dropout(p=dropout)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.norm(x)
        if exists(scale_shift):
            x += scale_shift[:, :, None, None]

        x = self.act(x)
        x = self.drop(x)
        x = self.proj(x)
        return x

class BlockNoGroup(nn.Module):
    def __init__(self, dim, dim_out, groups = 8, dropout=0.0):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.InstanceNorm2d(dim_out)
        self.drop = nn.Dropout(p=dropout)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.norm(x)
        if exists(scale_shift):
            x += scale_shift[:, :, None, None]

        x = self.act(x)
        x = self.drop(x)
        x = self.proj(x)
        return x    

    
class AdaptiveLayer(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.style_net = nn.Linear(style_dim, in_channel * 2)

        self.style_net.bias.data[:in_channel] = 1
        self.style_net.bias.data[in_channel:] = 0

    def forward(self, input, style = None):
        style = self.style_net(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = (gamma + 1) * input + beta

        return out

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, z_emb_dim = None, groups = 8, dropout=0.0, temp_nc = 3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups, temp_nc = temp_nc)
        self.block2 = Block(dim_out, dim_out, groups = groups, dropout=dropout, temp_nc = temp_nc)
        self.res_conv = nn.Conv2d(dim, dim_out, 1, groups=temp_nc) if dim != dim_out else nn.Identity()
        
        self.Dense_time = nn.Linear(time_emb_dim, dim)
        nn.init.zeros_(self.Dense_time.bias)
        
        self.adaptive = AdaptiveLayer(dim_out, z_emb_dim) if exists(z_emb_dim) else nn.Identity()

    def forward(self, x, time_emb = None, z_emb = None):
        time_input = self.Dense_time(time_emb)
        h = self.block1(x, scale_shift = time_input)
        if exists(z_emb):
            h = self.adaptive(h, z_emb)

        h = self.block2(h)
        return h + self.res_conv(x)
    

class ResnetBlockNoGroup(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, z_emb_dim = None, groups = 8, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = BlockNoGroup(dim, dim_out, groups = groups)
        self.block2 = BlockNoGroup(dim_out, dim_out, groups = groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1 ) if dim != dim_out else nn.Identity()
        
        self.Dense_time = nn.Linear(time_emb_dim, dim)
        # self.Dense_time.weight.data = default_init()(self.Dense_time.weight.data.shape)
        nn.init.zeros_(self.Dense_time.bias)
        
        self.adaptive = AdaptiveLayer(dim_out, z_emb_dim) if exists(z_emb_dim) else nn.Identity()

    def forward(self, x, time_emb = None, z_emb = None):

        # scale_shift = None
        # if exists(self.mlp) and exists(time_emb):
        #     time_emb = self.mlp(time_emb)
        #     time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        #     scale_shift = time_emb.chunk(2, dim = 1)
        
        time_input = self.Dense_time(time_emb)

        h = self.block1(x, scale_shift = time_input)

        if exists(z_emb):
            h = self.adaptive(h, z_emb)

        h = self.block2(h)

        return h + self.res_conv(x)    


class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash = flash)

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model

class UNet_feat_group(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        num_downs,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults = (1, 2, 2, 4),
        opt=None,
        use_dropout=False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = (False, False, False, True),
        flash_attn = False
    ):
        super().__init__()
        
        input_channels = input_nc
        self.dim = dim

        init_dim = default(init_dim, dim)
        
        self.init_pad = nn.ReflectionPad2d(3)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 0, groups=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        temp_nc = opt.temp_nc

        block_klass = partial(ResnetBlock, groups = resnet_block_groups, temp_nc = temp_nc)
        block_klass_no_group = partial(ResnetBlockNoGroup, groups = resnet_block_groups)
        
        dp = 0.0
        if use_dropout:
            dp = 0.1

        # time embeddings

        time_dim = dim * 4
        z_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        FullAttention = partial(Attention, flash = flash_attn)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim, z_emb_dim = z_dim, dropout=dp),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim, z_emb_dim = z_dim, dropout=dp),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_no_group(mid_dim, mid_dim, time_emb_dim = time_dim, z_emb_dim = z_dim, dropout=dp)
        self.mid_block1_wave = block_klass_no_group(mid_dim, mid_dim, time_emb_dim = time_dim, z_emb_dim = z_dim, dropout=dp)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = block_klass_no_group(mid_dim, mid_dim, time_emb_dim = time_dim, z_emb_dim = z_dim, dropout=dp)
        self.mid_block2_wave = block_klass_no_group(mid_dim, mid_dim, time_emb_dim = time_dim, z_emb_dim = z_dim, dropout=dp)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(nn.ModuleList([
                block_klass_no_group(dim_out + dim_in, dim_out, time_emb_dim = time_dim, z_emb_dim = z_dim, dropout=dp),
                block_klass_no_group(dim_out + dim_in, dim_out, time_emb_dim = time_dim, z_emb_dim = z_dim, dropout=dp),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        # default_out_dim = output_nc # channels * (1 if not learned_variance else 2)
        self.out_dim = output_nc # default(out_dim, default_out_dim)

        self.final_res_block = block_klass_no_group(dim * 2, dim, time_emb_dim = time_dim, z_emb_dim = z_dim)
        
        self.norm = nn.InstanceNorm2d(dim)
        self.act = nn.Tanh()
        self.final_pad = nn.ReflectionPad2d(3)
        self.final_conv = nn.Conv2d(dim, self.out_dim, kernel_size=7, padding=0)
        
        mapping_layers = [PixelNorm(),
                      nn.Linear(self.dim * 4, self.dim * 4),
                      nn.LeakyReLU(0.2)]
        for _ in range(opt.n_mlp):
            mapping_layers.append(nn.Linear(self.dim * 4, self.dim * 4))
            mapping_layers.append(nn.LeakyReLU(0.2))
        self.z_transform = nn.Sequential(*mapping_layers)
        
        self.dwt = DWT_2D("haar")
        self.iwt = IDWT_2D("haar")
        
        self.opt = opt

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, z, layers=[], encode_only=False):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        z_embed = self.z_transform(z)
        
        x = self.init_pad(x)
        x = self.init_conv(x)

        if z_embed.shape[0] == 2 and self.opt.phase == 'test':
            r = x.clone()
            r = torch.cat([r, r], dim=0)
        else:
            r = x.clone()

        t = self.time_mlp(time)

        h = []
        if len(layers) > 0:
            feat = x
            feats = [x]
            for block1, block2, downsample in self.downs:
                feat = block1(feat, t, z_embed)
                h.append(feat)

                feat = block2(feat, t, z_embed)
                h.append(feat)
                
                feats.append(feat)

                feat = downsample(feat)

            if len(feats) == len(layers) and encode_only:
                return feats
            
            x1 = self.mid_block1(feat, t, z_embed)
            xll, xlh, xhl, xhh = self.dwt(feat)
            xhh = self.mid_block1_wave(xhh, t, z_embed)
            x2 = self.iwt(xll, xlh, xhl, xhh)
            feat = x1 + x2

            feat = self.mid_attn(feat) + feat
            
            x1 = self.mid_block2(feat, t, z_embed)
            xll, xlh, xhl, xhh = self.dwt(feat)
            xhh = self.mid_block2_wave(xhh, t, z_embed)
            x2 = self.iwt(xll, xlh, xhl, xhh)
            feat = x1 + x2
            
            feats.append(feat)
            
            if len(feats) == len(layers) and encode_only:
                return feats

            for layer_id, (block1, block2, attn, upsample) in enumerate(self.ups):
                feat = torch.cat((feat, h.pop()), dim = 1)
                feat = block1(feat, t, z_embed)

                feat = torch.cat((feat, h.pop()), dim = 1)
                feat = block2(feat, t, z_embed)
                feat = attn(feat) + feat
                feats.append(feat)

                feat = upsample(feat)
                
                    
                if layer_id + len(feats) + 1 == len(layers) and encode_only:
                    return feats

            feat = torch.cat((feat, r), dim = 1)

            feat = self.final_res_block(feat, t)
            return feat, feats
            
        else:
            for block1, block2, downsample in self.downs:
                x = block1(x, t, z_embed)
                h.append(x)

                x = block2(x, t, z_embed)
                h.append(x)
                x = downsample(x)

            x1 = self.mid_block1(x, t, z_embed)
            xll, xlh, xhl, xhh = self.dwt(x)
            xhh = self.mid_block1_wave(xhh, t, z_embed)
            x2 = self.iwt(xll, xlh, xhl, xhh)
            x = x1 + x2
            
            x = self.mid_attn(x) + x
            
            x1 = self.mid_block2(x, t, z_embed)
            xll, xlh, xhl, xhh = self.dwt(x)
            xhh = self.mid_block2_wave(xhh, t, z_embed)
            x2 = self.iwt(xll, xlh, xhl, xhh)
            x = x1 + x2

            for block1, block2, attn, upsample in self.ups:
                x = torch.cat((x, h.pop()), dim = 1)
                x = block1(x, t, z_embed)

                x = torch.cat((x, h.pop()), dim = 1)
                x = block2(x, t, z_embed)
                x = attn(x) + x
                x = upsample(x)

            x = torch.cat((x, r), dim = 1)

            x = self.final_res_block(x, t, z_embed)
            return self.act(self.final_conv(self.final_pad(x)))

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

# main class

class Attend(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        flash = False
    ):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = AttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig(False, True, True)

    def flash_attn(self, q, k, v):
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p = self.dropout if self.training else 0.
            )

        return out

    def forward(self, q, k, v):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        if self.flash:
            return self.flash_attn(q, k, v)

        scale = q.shape[-1] ** -0.5

        # similarity

        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out
    
    
