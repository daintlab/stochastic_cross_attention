import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn
import torchvision

from torchvision.ops.misc import Conv2dNormActivation, MLP
from torchvision.transforms._presets import ImageClassification, InterpolationMode
from torchvision.utils import _log_api_usage_once
# from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._api import Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface

import numpy as np
from torch import einsum
from einops import rearrange, repeat

__all__ = [
    "VisionTransformer",
    "ViT_B_16_Weights",
    "vit_b_16",
]

class Attention_QKV(nn.Module):
    def __init__(self,dim,heads):
        super().__init__()
        dim_head = dim // heads
        inner_dim = dim_head *  heads
        self.heads = heads
        self.in_proj = nn.Linear(dim, inner_dim * 3)
    
    def forward(self,x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.in_proj(x).chunk(3, dim = -1)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        return q,k,v

class Attention_Out(nn.Module):
    def __init__(self,dim,heads,dropout = 0.):
        super().__init__()
        dim_head = dim // heads
        inner_dim = dim_head *  heads
        self.scale = dim_head ** -0.5
        self.out_proj = nn.Linear(inner_dim,dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,q,k,v):
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.dropout(self.out_proj(out))
        return out

class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d
    activation_layer: Callable[..., nn.Module] = nn.ReLU


class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        ca_prob:float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        # self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.self_attention_qkv = Attention_QKV(hidden_dim,num_heads)
        self.self_attention_out = Attention_Out(hidden_dim,num_heads,dropout=attention_dropout)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)
        
        # Probability to apply CA
        self.p = torch.tensor(ca_prob)

    def forward(self, input: torch.Tensor,key=None,val=None):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        
        q,k,v = self.self_attention_qkv(x)
        if key is not None and val is not None: # Train mode
            if torch.bernoulli(self.p): # Apply CA
                x = self.self_attention_out(q,key,val)
            else: # Apply SA
                x = self.self_attention_out(q,k,v)
        else: # Test mode
            x = self.self_attention_out(q,k,v)
            
        x = self.dropout(x)
        x = x + input
        
        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        ca_prob: list,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                ca_prob[i],
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor,key=None,val=None):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        input = self.dropout(input)
        
        for i, module in enumerate(self.layers):
            if key is None or val is None:
                input = module(input)
            else:
                input = module(input,key=key[i],val=val[i])
        return self.ln(input)

class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        ca_prob:list,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        conv_stem_configs: Optional[List[ConvStemConfig]] = None,
    ):
        super().__init__()
        _log_api_usage_once(self)
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        if conv_stem_configs is not None:
            # As per https://arxiv.org/abs/2106.14881
            seq_proj = nn.Sequential()
            prev_channels = 3
            for i, conv_stem_layer_config in enumerate(conv_stem_configs):
                seq_proj.add_module(
                    f"conv_bn_relu_{i}",
                    Conv2dNormActivation(
                        in_channels=prev_channels,
                        out_channels=conv_stem_layer_config.out_channels,
                        kernel_size=conv_stem_layer_config.kernel_size,
                        stride=conv_stem_layer_config.stride,
                        norm_layer=conv_stem_layer_config.norm_layer,
                        activation_layer=conv_stem_layer_config.activation_layer,
                    ),
                )
                prev_channels = conv_stem_layer_config.out_channels
            seq_proj.add_module(
                "conv_last", nn.Conv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1)
            )
            self.conv_proj: nn.Module = seq_proj
        else:
            self.conv_proj = nn.Conv2d(
                in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
            )

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1
        
        assert len(ca_prob) == num_layers
        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            ca_prob,
            norm_layer,
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor,key=None,val=None):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x,key,val)
        x = x[:, 0]
        x = self.heads(x)
        return x

        # Classifier "token" as used by standard language architectures
    


def _vision_transformer(
    patch_size: int,
    num_layers: int,
    num_heads: int,
    hidden_dim: int,
    mlp_dim: int,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> VisionTransformer:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        assert weights.meta["min_size"][0] == weights.meta["min_size"][1]
        _ovewrite_named_param(kwargs, "image_size", weights.meta["min_size"][0])
    image_size = kwargs.pop("image_size", 224)

    model = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        **kwargs,
    )

    if weights:
        ckpt = weights.get_state_dict(progress=progress)
        new_ckpt = {}
        for k,v in ckpt.items():
            if 'self_attention.in_proj_weight' in k:
                k = k.replace('self_attention.in_proj_weight','self_attention_qkv.in_proj.weight')
            elif 'self_attention.in_proj_bias' in k:
                k = k.replace('self_attention.in_proj_bias','self_attention_qkv.in_proj.bias')
            elif 'self_attention.out_proj' in k:
                k = k.replace('self_attention.out_proj','self_attention_out.out_proj')
            new_ckpt[k] = v
        model.load_state_dict(new_ckpt)
        # model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


_COMMON_META: Dict[str, Any] = {
    "categories": _IMAGENET_CATEGORIES,
}

_COMMON_SWAG_META = {
    **_COMMON_META,
    "recipe": "https://github.com/facebookresearch/SWAG",
    "license": "https://github.com/facebookresearch/SWAG/blob/main/LICENSE",
}


class ViT_B_16_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vit_b_16-c867db91.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 86567656,
            "min_size": (224, 224),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#vit_b_16",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.072,
                    "acc@5": 95.318,
                }
            },
            "_docs": """
                These weights were trained from scratch by using a modified version of `DeIT
                <https://arxiv.org/abs/2012.12877>`_'s training recipe.
            """,
        },
    )
    IMAGENET1K_SWAG_E2E_V1 = Weights(
        url="https://download.pytorch.org/models/vit_b_16_swag-9ac1b537.pth",
        transforms=partial(
            ImageClassification,
            crop_size=384,
            resize_size=384,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "num_params": 86859496,
            "min_size": (384, 384),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 85.304,
                    "acc@5": 97.650,
                }
            },
            "_docs": """
                These weights are learnt via transfer learning by end-to-end fine-tuning the original
                `SWAG <https://arxiv.org/abs/2201.08371>`_ weights on ImageNet-1K data.
            """,
        },
    )
    IMAGENET1K_SWAG_LINEAR_V1 = Weights(
        url="https://download.pytorch.org/models/vit_b_16_lc_swag-4e70ced5.pth",
        transforms=partial(
            ImageClassification,
            crop_size=224,
            resize_size=224,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "recipe": "https://github.com/pytorch/vision/pull/5793",
            "num_params": 86567656,
            "min_size": (224, 224),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.886,
                    "acc@5": 96.180,
                }
            },
            "_docs": """
                These weights are composed of the original frozen `SWAG <https://arxiv.org/abs/2201.08371>`_ trunk
                weights and a linear classifier learnt on top of them trained on ImageNet-1K data.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1



# @register_model()
@handle_legacy_interface(weights=("pretrained", ViT_B_16_Weights.IMAGENET1K_V1))
def vit_b_16(*, weights: Optional[ViT_B_16_Weights] = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_b_16 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.
    Args:
        weights (:class:`~torchvision.models.ViT_B_16_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_B_16_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.ViT_B_16_Weights
        :members:
    """
    weights = ViT_B_16_Weights.verify(weights)

    return _vision_transformer(
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        weights=weights,
        progress=progress,
        **kwargs,
    )
