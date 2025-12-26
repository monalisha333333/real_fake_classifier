import torch.nn as nn
from torch import Tensor
import math
import torch
from typing import *

class ESPCN(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            channels: int,
            upscale_factor: int,
    ) -> None:
        super().__init__()
        hidden_channels = channels // 2
        out_channels = int(out_channels * (upscale_factor ** 2))
        self.bn = nn.BatchNorm2d(in_channels)
        # Feature mapping
        self.feature_maps = nn.Sequential(
            nn.Conv2d(in_channels, channels, (5, 5), (1, 1), (2, 2)),
            nn.Tanh(),
            nn.Conv2d(channels, hidden_channels, (3, 3), (1, 1), (1, 1)),
            nn.Tanh(),
        )

        # Sub-pixel convolution layer
        self.sub_pixel_0 = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(upscale_factor),
            # nn.Sigmoid(),
        )

        # Initial model weights
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if module.in_channels == 32:
                    nn.init.normal_(module.weight.data,
                                    0.0,
                                    0.001)
                    nn.init.zeros_(module.bias.data)
                else:
                    nn.init.normal_(module.weight.data,
                                    0.0,
                                    math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                    nn.init.zeros_(module.bias.data)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function.
    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.bn(x)
        x = self.feature_maps(x)
        x_h = self.sub_pixel_0(x)

        return x_h

class fft_filter(nn.Module):
    def __init__(self, radiuslow=35, radiushigh=120, rows=384, cols=384):
        super(fft_filter, self).__init__()
        self.radiuslow = radiuslow
        self.radiushigh = radiushigh
        self.rows = rows
        self.cols = cols
        # preset masks M_{mid}
        self.register_buffer('i_mask', self.init_mask())
        # # encode and decode frequency mask
        self.mask_autoencoder = ESPCN(in_channels=3, out_channels=1, channels=64, upscale_factor=1)
        for m in self.mask_autoencoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def init_mask(self):
        mask = torch.ones((1, self.rows, self.cols), dtype=torch.float32, requires_grad=False)
        crow, ccol = self.rows // 2 , self.cols // 2
        center = [crow, ccol]
        x, y = torch.meshgrid(torch.arange(self.rows), torch.arange(self.cols), indexing='ij')
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 < self.radiuslow*self.radiuslow
        mask[:, mask_area] = 0
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 >= self.radiushigh*self.radiushigh
        mask[:, mask_area] = 0
        return mask
    

    # implement frequency filter
    def middle_pass_filter(self, image, return_mask=False, debug=False):
        freq_image = torch.fft.fftn(image * 255, dim=(-2, -1))
        freq_image = torch.fft.fftshift(freq_image, dim=(-2, -1))
        delta_mask = self.mask_autoencoder(
            (20*torch.log(torch.abs(freq_image)+1e-7))/255
            )
        delta_mask = torch.sigmoid(delta_mask)
        # Ensure shapes match
        i_mask = self.i_mask.unsqueeze(0) if self.i_mask.dim() == 3 else self.i_mask
        i_mask = i_mask.to(delta_mask.device)
        # Residual learning + bounding
        mask_mid_frq = torch.sigmoid(i_mask + delta_mask)
        mask_mid_frq = mask_mid_frq.to(freq_image.dtype)

        middle_freq = freq_image * mask_mid_frq
        middle_freq1 = torch.fft.ifftshift(middle_freq, dim=(-2, -1))
        masked_image_array = torch.fft.ifftn(middle_freq1, dim=(-2, -1))
        z = torch.abs(masked_image_array)
        _min = torch.min(z)
        _max = torch.max(z)
        middle_freq_image = (z)/(_max-_min)
        
        if debug:
            self.debug_tensors = {
                "image": image.detach().cpu(),
                "fft_mag": torch.log(torch.abs(freq_image) + 1e-8).detach().cpu(),
                "i_mask": i_mask.detach().cpu(),
                "delta_mask": delta_mask.detach().cpu(),
                "mask_mid_frq": mask_mid_frq.detach().cpu(),
                "middle_freq_mag": torch.log(torch.abs(middle_freq) + 1e-8).detach().cpu(),
                "output": middle_freq_image.detach().cpu(),
            }

        if return_mask:
            return middle_freq_image, mask_mid_frq 
        else:
            return middle_freq_image
    
    
    def forward(self, image, return_mask=False, debug=False):

        return self.middle_pass_filter(image,return_mask, debug)