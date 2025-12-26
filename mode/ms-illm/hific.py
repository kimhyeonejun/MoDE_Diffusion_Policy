# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# MS-ILLM의 hyper parameter 세팅 그대로 가지고 데이터셋만 바꿔서 학습 해보기 (training step 등)

from typing import List, NamedTuple, Optional, Tuple, Union
import logging

import torch
from torch import Tensor

import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

try:
    from ._hyperprior_autoencoder import _HyperpriorAutoencoderBase
except ImportError:
    from _hyperprior_autoencoder import _HyperpriorAutoencoderBase

from typing import List, Tuple

import torch.nn.functional as F


def pad_image_to_factor(
    image: Tensor, factor: int, mode: str = "reflect"
) -> Tuple[Tensor, Tuple[int, int]]:
    
    # pad image if it's not divisible by downsamples
    _, _, height, width = image.shape
    pad_height = (factor - (height % factor)) % factor
    pad_width = (factor - (width % factor)) % factor
    if pad_height != 0 or pad_width != 0:
        image = F.pad(image, (0, pad_width, 0, pad_height), mode=mode)

    return image, (height, width)

class _ResidualBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.sequence = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride, padding=padding),
            _channel_norm_2d(channels, affine=True),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size, stride, padding=padding),
            _channel_norm_2d(channels, affine=True),
        )

    def forward(self, x):
        return x + self.sequence(x)

class ChannelNorm2D(nn.Module):
    def __init__(self, input_channels: int, epsilon: float = 1e-3, affine: bool = True):
        super().__init__()

        if input_channels <= 1:
            raise ValueError(
                "ChannelNorm only valid for channel counts greater than 1."
            )

        self.epsilon = epsilon
        self.affine = affine

        if affine is True:
            self.gamma = nn.Parameter(torch.ones(1, input_channels, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, input_channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        mean = torch.mean(x, dim=1, keepdim=True)
        variance = torch.var(x, dim=1, keepdim=True)

        x_normed = (x - mean) * torch.rsqrt(variance + self.epsilon)

        if self.affine is True:
            x_normed = self.gamma * x_normed + self.beta

        return x_normed

def _channel_norm_2d(input_channels, affine=True):
    return ChannelNorm2D(
        input_channels,
        affine=affine,
    )

class HyperpriorOutput(NamedTuple):
    image: Tensor
    latent: Tensor
    latent_likelihoods: Tensor
    quantized_latent_likelihoods: Tensor
    hyper_latent: Tensor
    hyper_latent_likelihoods: Tensor
    quantized_hyper_latent_likelihoods: Tensor
    quantized_latent: Optional[Tensor] = None
    quantized_hyper_latent: Optional[Tensor] = None


class HyperpriorCompressedOutput(NamedTuple):
    latent_strings: Union[List[str], List[List[str]]]
    hyper_latent_strings: List[str]
    image_size: Tuple[int, int]
    padded_size: Tuple[int, int]


def _channel_norm_2d(input_channels, affine=True):
    return ChannelNorm2D(
        input_channels,
        affine=affine,
    )

# injector, extractor
class Injector(nn.Module) : # text 정보를 image에 주입
    def __init__(self, dim, text_feature_dim ,num_attn_head=6) :
        super().__init__()

        self.text_feature_dim_proj = nn.Linear(text_feature_dim, dim)
        nn.init.kaiming_normal_(self.text_feature_dim_proj.weight)

        self.image_norm = nn.LayerNorm(dim)
        self.text_norm = nn.LayerNorm(dim)

        self.cross_attn = nn.MultiheadAttention(dim, num_attn_head, batch_first=True)
        self.gamma = nn.Parameter(0.0 * torch.ones((dim)), requires_grad=True) # balance the attention layer’s output and the input feature Fi , which is initialized with 0
    
    def forward(self, image_features:torch.Tensor, text_features:torch.Tensor, get_norm_and_attn_map:bool) :    
        
        b,c,h,w = image_features.size()
        image_features = image_features.contiguous().flatten(2).permute(0,2,1) # token처럼

        text_features = self.text_feature_dim_proj(text_features)

        text_features = self.text_norm(text_features)
        image_features = self.image_norm(image_features)                                                                                                                                                                          

        attn_out, attn_weights = self.cross_attn(image_features, text_features, text_features)
        
        if get_norm_and_attn_map == True:
            norm_image = image_features.norm().item()
            norm_text = (self.gamma * attn_out).norm().item()

        image_features = image_features + self.gamma * attn_out
        image_features = image_features.contiguous().permute(0,2,1).view(b,c,h,w)

        if get_norm_and_attn_map == True:
            return image_features, attn_weights, norm_image, norm_text
        elif get_norm_and_attn_map == False:
            return image_features, None, None, None

class Extractor(nn.Module) : # image 정보를 text에 주입(image를 고려한 text 정보 추출 후 injector에 넣어주는 구조)
    def __init__(self, dim, text_feature_dim, num_attn_head=6) :
        super().__init__()

        self.image_feature_dim_proj = nn.Linear(dim, text_feature_dim)
        nn.init.kaiming_normal_(self.image_feature_dim_proj.weight)

        self.image_norm = nn.LayerNorm(text_feature_dim)
        self.text_norm = nn.LayerNorm(text_feature_dim)
        self.cross_attn = nn.MultiheadAttention(text_feature_dim, num_attn_head, batch_first=True)
    
    def forward(self, text_features:torch.Tensor, image_features:torch.Tensor, get_norm_and_attn_map:bool) :

        image_features = image_features.contiguous().flatten(2).permute(0,2,1) # token처럼

        image_features = self.image_feature_dim_proj(image_features)

        text_features = self.text_norm(text_features)
        image_features = self.image_norm(image_features)

        attn_out, attn_weights = self.cross_attn(text_features, image_features, image_features)
        
        if get_norm_and_attn_map == True:
            norm_image = attn_out.norm().item()
            norm_text = text_features.norm().item()
        
        text_features = text_features + attn_out
        
        if get_norm_and_attn_map == True:
            return text_features, attn_weights, norm_image, norm_text
        elif get_norm_and_attn_map == False:
            return text_features, None, None, None

class HiFiCEncoder(torch.nn.Module):
    """
    High-Fidelity Generative Image Compression (HiFiC) encoder.

    Args:
        input_dimensions: shape of the input tensor
        latent_features: number of bottleneck features
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_features: int = 220,
    ):
        super().__init__()

        blocks: List[nn.Module] = []
        for index, out_channels in enumerate((60, 120, 240, 480, 960)):
            if index == 0:
                block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3),
                    _channel_norm_2d(out_channels, affine=True),
                    nn.ReLU(),
                )
            else:
                block = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    _channel_norm_2d(out_channels, affine=True),
                    nn.ReLU(),
                )

            in_channels = out_channels
            blocks += [block]

        blocks += [nn.Conv2d(out_channels, latent_features, kernel_size=3, padding=1)]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)

class HiFiCGenerator(torch.nn.Module):
    """
    High-Fidelity Generative Image Compression (HiFiC) generator.

    Args:
        input_dimensions: shape of the input tensor
        batch_size: number of images per batch
        latent_features: number of bottleneck features
        n_residual_blocks: number of residual blocks
    """

    def __init__(
        self,
        image_channels: int = 3,
        latent_features: int = 220,
        n_residual_blocks: int = 9,
    ):
        super(HiFiCGenerator, self).__init__()

        self.n_residual_blocks = n_residual_blocks

        filters = [960, 480, 240, 120, 60]

        self.block_0 = nn.Sequential(
            _channel_norm_2d(latent_features, affine=True),
            nn.Conv2d(latent_features, filters[0], kernel_size=3, padding=1),
            _channel_norm_2d(filters[0], affine=True),
        )

        resid_blocks = []
        for _ in range(self.n_residual_blocks):
            resid_blocks.append(_ResidualBlock((filters[0])))

        self.resid_blocks = nn.Sequential(*resid_blocks)

        blocks: List[nn.Module] = []
        in_channels = filters[0]
        for out_channels in filters[1:]:
            blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        output_padding=1,
                        stride=2,
                        padding=1,
                    ),
                    _channel_norm_2d(out_channels, affine=True),
                    nn.ReLU(),
                )
            )

            in_channels = out_channels

        blocks.append(
            nn.Conv2d(
                filters[-1], out_channels=image_channels, kernel_size=7, padding=3
            )
        )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        x = self.block_0(x)
        x = x + self.resid_blocks(x)

        return self.blocks(x)


def _conv(
    cin: int,
    cout: int,
    kernel_size: int = 5,
    stride: int = 2,
) -> nn.Conv2d:
    return nn.Conv2d(
        cin, cout, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2
    )


def _deconv(
    cin: int,
    cout: int,
    kernel_size: int = 5,
    stride: int = 2,
) -> nn.ConvTranspose2d:
    return nn.ConvTranspose2d(
        cin,
        cout,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=1,
        padding=kernel_size // 2,
    )

class HiFiCAutoencoder(_HyperpriorAutoencoderBase):

    def __init__(
        self,
        in_channels: int = 3,
        latent_features: int = 220,
        hyper_features: int = 320,
        num_residual_blocks: int = 9,
        freeze_encoder: bool = False,
        freeze_bottleneck: bool = False,
    ):
        super().__init__()
        self._factor = 64  # this is the full downsampling factor
        self._frozen_encoder = False
        self._frozen_bottleneck = False
        self.logger = logging.getLogger(self.__class__.__name__)
        self.encoder = HiFiCEncoder(
            in_channels=in_channels, latent_features=latent_features
        )
        self.decoder = HiFiCGenerator(
            image_channels=in_channels,
            latent_features=latent_features,
            n_residual_blocks=num_residual_blocks,
        )
        self.hyper_analysis = nn.Sequential(
            _conv(latent_features, hyper_features, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            _conv(hyper_features, hyper_features),
            nn.ReLU(inplace=True),
            _conv(hyper_features, hyper_features),
        )
        self.hyper_synthesis_mean = nn.Sequential(
            _deconv(hyper_features, hyper_features),
            nn.ReLU(inplace=True),
            _deconv(hyper_features, hyper_features),
            nn.ReLU(inplace=True),
            _conv(
                hyper_features,
                latent_features,
                stride=1,
                kernel_size=3,
            ),
        )
        self.hyper_synthesis_scale = nn.Sequential(
            _deconv(hyper_features, hyper_features),
            nn.ReLU(inplace=True),
            _deconv(hyper_features, hyper_features),
            nn.ReLU(inplace=True),
            _conv(
                hyper_features,
                latent_features,
                stride=1,
                kernel_size=3,
            ),
        )
        self.hyper_bottleneck = EntropyBottleneck(hyper_features)
        self.latent_bottleneck = GaussianConditional(scale_table=None)

        if freeze_encoder:
            self.freeze_encoder()
        if freeze_bottleneck:
            self.freeze_bottleneck()

        self.set_compress_cpu_layers(
            [
                self.hyper_bottleneck,
                self.latent_bottleneck,
                self.hyper_synthesis_scale,
                self.hyper_synthesis_mean,
                self.hyper_analysis,
            ]
        )

    @property
    def factor(self) -> int:
        return self._factor

    @property
    def frozen_encoder(self) -> bool:
        return self._frozen_encoder

    @property
    def frozen_bottleneck(self) -> bool:
        return self._frozen_bottleneck

    def update_tensor_devices(self, target_operation: str):
        """
        Updates location of model weights based on target_operation.

        Args:
            target_operation: Either ''forward'' or ''compress''. For
                ''forward'', all tensors will be located on the model device.
                For ''compress'', weights that are used for likelihood
                calculation will be held on the CPU.
        """
        if target_operation not in ("forward", "compress"):
            raise ValueError("Target operation must be 'forward' or 'compress'.")

        if target_operation == "forward":
            target_device = self.encoder.blocks[0][0].weight.device
            self.to(target_device)
        else:
            self.update()
            self._set_devices_for_compress()

        self._device_setting = target_operation

    def _set_devices_for_compress(self):
        cpu = torch.device("cpu")
        self.hyper_bottleneck.to(cpu)
        self.latent_bottleneck.to(cpu)
        self.hyper_synthesis_scale.to(cpu)
        self.hyper_synthesis_mean.to(cpu)
        self.hyper_analysis.to(cpu)

    def _check_compress_devices(self) -> bool:
        result = True
        cpu = torch.device("cpu")
        for module in [
            self.hyper_bottleneck,
            self.latent_bottleneck,
            self.hyper_analysis,
            self.hyper_synthesis_mean,
            self.hyper_synthesis_scale,
        ]:
            for param in module.parameters():
                if not param.device == cpu:
                    result = False

        return result

    def freeze_encoder(self):
        """Freeze and disable training for the encoder."""
        self._frozen_encoder = True
        self.logger.info("Freezing encoder!")
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad_(False)

    def freeze_bottleneck(self):
        """Freeze and disable training for the bottleneck."""
        module: nn.Module
        self._frozen_bottleneck = True
        self.logger.info("Freezing bottleneck!")
        for module in [
            self.hyper_bottleneck,
            self.hyper_synthesis_mean,
            self.hyper_synthesis_scale,
            self.hyper_analysis,
            self.latent_bottleneck,
        ]:
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)

    def train(self, mode: bool = True) -> "HiFiCAutoencoder":
        model = super().train(mode)
        if model._frozen_encoder:
            model.freeze_encoder()
        if model._frozen_bottleneck:
            model.freeze_bottleneck()

        return model

    def forward(self, image: Tensor) -> HyperpriorOutput:
        if not self.device_setting == "forward":
            raise RuntimeError(
                "Must call update_tensor_devices('forward') to use model forward."
            )

        # encode image to latent
        latent = self.encoder(image)

        # bottleneck processing
        hyper_latent = self.hyper_analysis(latent)
        hyper_latent, hyper_likelihoods = self.hyper_bottleneck(hyper_latent)
        with torch.no_grad():
            _, quantized_hyper_latent_likelihoods = self.hyper_bottleneck(
                hyper_latent, training=False
            )
        means = self.hyper_synthesis_mean(hyper_latent)
        scales = self.hyper_synthesis_scale(hyper_latent)
        quantized_latents, latent_likelihoods = self.latent_bottleneck(
            latent, scales, means=means
        )
        with torch.no_grad():
            _, quantized_latent_likelihoods = self.latent_bottleneck(
                latent, scales, means=means, training=False
            )
        if self.training:
            # we use straight-through to train the generator
            latent = self._ste_quantize(latent, means)
        else:
            # means we're in eval mode and the latents have been rounded
            latent = quantized_latents

        # reconstruct the image
        reconstruction = self.decoder(latent)

        return HyperpriorOutput(
            image=reconstruction,
            latent=latent,
            latent_likelihoods=latent_likelihoods,
            quantized_latent_likelihoods=quantized_latent_likelihoods.detach(),
            hyper_latent=hyper_latent,
            hyper_latent_likelihoods=hyper_likelihoods,
            quantized_hyper_latent_likelihoods=quantized_hyper_latent_likelihoods.detach(),
        )

    def compress(
        self, image: Tensor, force_cpu: bool = True
    ) -> HyperpriorCompressedOutput:
        """
        Compress a batch of images into strings.

        Args:
            image: Tensor to compress (in [0, 1] floating point range).
            force_cpu: Whether to throw an error if any operations are not on
                CPU.
        """
        if not self._on_cpu():
            if force_cpu:
                raise ValueError("All params must be on CPU if force_cpu=True.")

            if not self._check_compress_devices():
                raise ValueError(
                    "Some layers on GPU that should be on CPU. Call "
                    "update_tensor_devices('compress') to use partial-GPU "
                    "compression."
                )

        image, (height, width) = pad_image_to_factor(image, self._factor)

        # image analysis
        latent = self.encoder(image).cpu()

        # hyper analysis
        hyper_latent = self.hyper_analysis(latent)

        # hyper bottleneck
        hyper_latent_strings = self.hyper_bottleneck.compress(hyper_latent)
        hyper_latent_decoded = self.hyper_bottleneck.decompress(
            hyper_latent_strings, hyper_latent.shape[-2:]
        )

        # hyper synthesis
        means = self.hyper_synthesis_mean(hyper_latent_decoded)
        scales = self.hyper_synthesis_scale(hyper_latent_decoded)

        # latent compression
        indexes = self.latent_bottleneck.build_indexes(scales)
        latent_strings = self.latent_bottleneck.compress(latent, indexes, means=means)

        return HyperpriorCompressedOutput(
            latent_strings=latent_strings,
            hyper_latent_strings=hyper_latent_strings,
            image_size=(height, width),
            padded_size=(image.shape[-2], image.shape[-1]),
        )

    def decompress(
        self, compressed_data: HyperpriorCompressedOutput, force_cpu: bool = True
    ) -> Tensor:
        """
        Decompress a batch of images from strings.

        Args:
            compressed_data: Strings of data to decompress.
            force_cpu: Whether to throw an error if any operations are not on
                CPU.
        """
        if not self._on_cpu():
            if force_cpu:
                raise ValueError("All params must be on CPU if force_cpu=True.")

            if not self._check_compress_devices():
                raise ValueError(
                    "Some layers on GPU that should be on CPU. Call "
                    "update_tensor_devices('compress') to use partial-GPU "
                    "compression."
                )

            device = self.decoder.blocks[-1].weight.device
        else:
            device = torch.device("cpu")

        latent_size = (
            compressed_data.padded_size[0] // 2**4,
            compressed_data.padded_size[1] // 2**4,
        )
        hyper_latent_size = (latent_size[0] // 2**2, latent_size[1] // 2**2)

        # hyper synthesis
        hyper_latent_decoded = self.hyper_bottleneck.decompress(
            compressed_data.hyper_latent_strings, hyper_latent_size
        )
        means = self.hyper_synthesis_mean(hyper_latent_decoded)
        scales = self.hyper_synthesis_scale(hyper_latent_decoded)

        # image synthesis
        indexes = self.latent_bottleneck.build_indexes(scales)
        latent_decoded = self.latent_bottleneck.decompress(
            compressed_data.latent_strings, indexes, means=means
        )
        reconstruction: Tensor = self.decoder(latent_decoded.to(device))

        return reconstruction[
            :, :, : compressed_data.image_size[0], : compressed_data.image_size[1]
        ].clamp(0.0, 1.0)