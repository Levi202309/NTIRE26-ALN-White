import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

# A ConvNet for the 2020s 
# original implementation  https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
# paper https://arxiv.org/pdf/2201.03545.pdf

class ConvNeXt0(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, block, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    def __init__(self, block, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 27, 3], dims=[256, 512, 1024,2048], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _adjust_size(self, source, direction, operation):
        """
        调整特征图尺寸,每次固定调整1像素
    
        Args:
            source (Tensor): 输入特征图，形状为 (B, C, H, W)
            direction (str): 调整方向，'h'（高度方向）或 'w'（宽度方向）
            operation (str): 操作类型，'pad'（填充）或 'crop'（裁剪）
    
        Returns:
            Tensor: 调整后的特征图
        """
        if operation == 'pad':
            # 填充逻辑：在指定方向的最下方或最右侧填充1像素
            if direction == 'h':
                # 高度方向填充：在底部加1行 (padding格式：左, 右, 上, 下)
                source = F.pad(source, (0, 0, 0, 1))  
            elif direction == 'w':
                # 宽度方向填充：在右侧加1列
                source = F.pad(source, (0, 1, 0, 0))  
        elif operation == 'crop':
            # 裁剪逻辑：在指定方向裁剪最后1像素
            _, _, h, w = source.size()
            if direction == 'h':
                # 高度方向裁剪：移除底部1行
                source = source[:, :, :h-1, :]
            elif direction == 'w':
                # 宽度方向裁剪：移除右侧1列
                source = source[:, :, :, :w-1]
        else:
            raise ValueError("operation必须是'pad'或'crop'")
        return source

    def forward(self, x):  # (1,3,1000,750)
        # (1,3,750,1000)
        x = self._adjust_size(x, direction='h', operation='pad')
        # (1,3,751,1000)
        x = self._adjust_size(x, direction='h', operation='pad')
        # (1,3,752,1000)
        x_layer1 = self.downsample_layers[0](x)  # (1,256,250,187)
        # (1,256,188,250)
        x_layer1 = self.stages[0](x_layer1)  # (1,256,250,187)
        # (1,256,188,250)
        x_layer2 = self.downsample_layers[1](x_layer1)  # (1,512,125,93)
        # (1,512,94,125)
        x_layer2 = self.stages[1](x_layer2)  # (1,512,125,93)
        # (1,512,94,125)
        x_layer2_even = x_layer2
        # (1,512,94,125)
        x_layer2_even = self._adjust_size(x_layer2_even, direction='w', operation='pad')
        # (1,512,94,126)
        # x_layer3 = self.downsample_layers[2](x_layer2)  # (1,1024,62,46)
        x_layer3 = self.downsample_layers[2](x_layer2_even)  # (1,1024,62,46)
        # (1,1024,47,63)
        out = self.stages[2](x_layer3)  # (1,1024,62,46)
        # (1,1024,47,63)

        return x_layer1, x_layer2, out
        # (1,256,188,250),(1,512,94,125),(1,1024,47,63)
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class CP_Attention_block(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(CP_Attention_block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)
    def forward(self, x):  # (1,1024,62,46)
        res = self.act1(self.conv1(x))  # (1,1024,62,46)
        res = res + x  # (1,1024,62,46)
        res = self.conv2(res)  # (1,1024,62,46)
        res = self.calayer(res)  # (1,1024,62,46)
        res = self.palayer(res)  # (1,1024,62,46)
        res += x
        return res

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class knowledge_adaptation_convnext(nn.Module):
    def __init__(self):
        super(knowledge_adaptation_convnext, self).__init__()
        self.encoder = ConvNeXt(Block, in_chans=3,num_classes=1000, depths=[3, 3, 27, 3], dims=[256, 512, 1024,2048], drop_path_rate=0., layer_scale_init_value=1e-6, head_init_scale=1.)
        pretrained_model = ConvNeXt0(Block, in_chans=3,num_classes=1000, depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], drop_path_rate=0., layer_scale_init_value=1e-6, head_init_scale=1.)
        #pretrained_model=nn.DataParallel(pretrained_model)
        # checkpoint=torch.load('./weights/convnext_xlarge_22k_1k_384_ema.pth')

        # device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        # checkpoint=torch.load('/root/autodl-tmp/ckpt/IFBlend_convnext_xlarge_22k_1k_384_ema.pth',map_location=device)
        checkpoint=torch.load('/root/autodl-tmp/ckpt/IFBlend_convnext_xlarge_22k_1k_384_ema.pth')

        #for k,v in checkpoint["model"].items():
            #print(k)
        #url="https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_384.pth"
        
        #checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cuda:0")
        pretrained_model.load_state_dict(checkpoint["model"])
        
        pretrained_dict = pretrained_model.state_dict()
        model_dict = self.encoder.state_dict()
        key_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(key_dict)
        self.encoder.load_state_dict(model_dict)


        self.up_block= nn.PixelShuffle(2)
        self.attention0 = CP_Attention_block(default_conv, 1024, 3)
        self.attention1 = CP_Attention_block(default_conv, 256, 3)
        self.attention2 = CP_Attention_block(default_conv, 192, 3)
        self.attention3 = CP_Attention_block(default_conv, 112, 3)
        self.attention4 = CP_Attention_block(default_conv, 28, 3)
        self.conv_process_1 = nn.Conv2d(28, 28, kernel_size=3,padding=1)
        self.conv_process_2 = nn.Conv2d(28, 28, kernel_size=3,padding=1)
        self.tail = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(28, 3, kernel_size=7, padding=0), nn.Tanh())
    
    def _adjust_size(self, source, direction, operation):
        """
        调整特征图尺寸,每次固定调整1像素
    
        Args:
            source (Tensor): 输入特征图，形状为 (B, C, H, W)
            direction (str): 调整方向，'h'（高度方向）或 'w'（宽度方向）
            operation (str): 操作类型，'pad'（填充）或 'crop'（裁剪）
    
        Returns:
            Tensor: 调整后的特征图
        """
        if operation == 'pad':
            # 填充逻辑：在指定方向的最下方或最右侧填充1像素
            if direction == 'h':
                # 高度方向填充：在底部加1行 (padding格式：左, 右, 上, 下)
                source = F.pad(source, (0, 0, 0, 1))  
            elif direction == 'w':
                # 宽度方向填充：在右侧加1列
                source = F.pad(source, (0, 1, 0, 0))  
        elif operation == 'crop':
            # 裁剪逻辑：在指定方向裁剪最后1像素
            _, _, h, w = source.size()
            if direction == 'h':
                # 高度方向裁剪：移除底部1行
                source = source[:, :, :h-1, :]
            elif direction == 'w':
                # 宽度方向裁剪：移除右侧1列
                source = source[:, :, :, :w-1]
        else:
            raise ValueError("operation必须是'pad'或'crop'")
        return source

    def forward(self, input):  # (1,3,1000,750)
        x_layer1, x_layer2, x_output = self.encoder(input)  # (1,256,250,187),(1,512,125,93),(1,1024,62,46)
        # (1,256,188,250),(1,512,94,125),(1,1024,47,63)
        x_mid = self.attention0(x_output)  #[1024,24,24]  # (1,1024,62,46)
        # (1,1024,47,63)
        x = self.up_block(x_mid)      #[256,48,48]  # (1,256,124,92)
        # (1,256,94,126)
        x = self.attention1(x)  
        # (1,256,94,126)
        x = self._adjust_size(x, direction='w', operation='crop')
        # (1,256,94,125)
        x = torch.cat((x, x_layer2), 1)  #[768,48,48]  # (1,768,124,92)
        # (1,768,94,125)
        x = self.up_block(x)            #[192,96,96]  # (1,192,248,184)
        # (1,192,188,250)
        x = self.attention2(x)
        # (1,192,188,250)
        x = torch.cat((x, x_layer1), 1)   #[448,96,96]  # (1,448,248,184)
        # (1,448,188,250)
        x = self.up_block(x)            #[112,192,192]  # (1,112,124,92)
        # (1,112,376,500)
        x = self.attention3(x)              
        # (1,112,376,500)
        x = self._adjust_size(x, direction='h', operation='crop')
        # (1,112,375,500)
        x = self.up_block(x)        #[28,384,384]  # (1,64,248,184)
        # (1,28,750,1000)
        x = self.attention4(x)
        # (1,28,750,1000)
        x=self.conv_process_1(x)
        out=self.conv_process_2(x)
        return out

