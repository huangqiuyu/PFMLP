
import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class MlpHead(nn.Module):
    def __init__(self, dim, num_classes=1000, mlp_ratio=4):
        super().__init__()
        hidden_features = min(2048, int(mlp_ratio * dim))
        self.fc1 = nn.Linear(dim, hidden_features, bias=False)
        self.norm = nn.BatchNorm1d(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, groups=1, kernel_size=3, stride=1, use_act=True):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, groups=groups, padding=kernel_size//2, bias=False)
        self.norm = nn.BatchNorm2d(out_planes)
        self.act = nn.GELU() if use_act else nn.Identity()

    def forward(self, x):
        out = self.norm(self.conv(x))
        out = self.act(out)
        return out


class LearnablePool2d(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.dim = dim
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.Tensor(1, 1, kernel_size, kernel_size), requires_grad=True)
        nn.init.normal_(self.weight, 0, 0.01)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        weight = self.weight.repeat(self.dim, 1, 1, 1)
        out = nn.functional.conv2d(x, weight, None, self.stride, self.padding, groups=self.dim)
        return self.norm(out)


class ChannelLearnablePool2d(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride, groups=dim, padding=padding, bias=False)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        out = self.conv(x)
        return self.norm(out)


class PyramidFC(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, use_dw=False):
        super(PyramidFC, self).__init__()
        if use_dw:
            block = ChannelLearnablePool2d
        else:
            block = LearnablePool2d

        self.branch_1 = nn.Sequential(
            block(inplanes, kernel_size=3, stride=1, padding=1),
            ConvX(inplanes, planes, groups=1, kernel_size=1, use_act=False)
        )
        self.branch_2 = nn.Sequential(
            block(inplanes, kernel_size=5, stride=2, padding=2),
            ConvX(inplanes, planes, groups=1, kernel_size=1, use_act=False)
        )
        self.branch_3 = nn.Sequential(
            block(inplanes, kernel_size=7, stride=3, padding=3),
            ConvX(inplanes, planes, groups=1, kernel_size=1, use_act=False)
        )
        self.branch_4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvX(inplanes, planes, groups=1, kernel_size=1, use_act=False)
        )
        self.act = nn.GELU()

    def forward(self, x):
        b, c, h, w = x.shape
        x1 = self.branch_1(x)
        x2 = nn.functional.interpolate(self.branch_2(x), size=(h, w), scale_factor=None, mode='nearest')
        x3 = nn.functional.interpolate(self.branch_3(x), size=(h, w), scale_factor=None, mode='nearest')
        x4 = self.branch_4(x)
        out = self.act(x1 + x2 + x3 + x4)
        return out


class BottleNeck(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, expand_ratio=1.0, mlp_ratio=1.0, use_dw=False, drop_path=0.0):
        super(BottleNeck, self).__init__()
        if use_dw:
            block = ChannelLearnablePool2d
        else:
            block = LearnablePool2d
        expand_planes = int(in_planes*expand_ratio)
        mid_planes = int(out_planes*mlp_ratio)

        self.smlp = nn.Sequential(
            PyramidFC(in_planes, expand_planes, kernel_size=3, stride=stride, use_dw=use_dw),
            ConvX(expand_planes, in_planes, groups=1, kernel_size=1, stride=1, use_act=False)
        )
        self.cmlp = nn.Sequential(
            ConvX(in_planes, mid_planes, groups=1, kernel_size=1, stride=1, use_act=True),
            block(mid_planes, kernel_size=3, stride=stride, padding=1) if stride==1 else ConvX(mid_planes, mid_planes, groups=mid_planes, kernel_size=3, stride=2, use_act=False),
            ConvX(mid_planes, out_planes, groups=1, kernel_size=1, stride=1, use_act=False)
        )

        self.skip = nn.Identity()
        if stride == 2 and in_planes != out_planes:
            self.skip = nn.Sequential(
                ConvX(in_planes, in_planes, groups=in_planes, kernel_size=3, stride=2, use_act=False),
                ConvX(in_planes, out_planes, groups=1, kernel_size=1, stride=1, use_act=False)
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = self.drop_path(self.smlp(x)) + x
        x = self.drop_path(self.cmlp(x)) + self.skip(x)
        return x


class PFMLP(nn.Module):
    # pylint: disable=unused-variable
    def __init__(self, dims, layers, block=BottleNeck, expand_ratio=1.0, mlp_ratio=1.0, use_dw=False, drop_path_rate=0., num_classes=1000):
        super(PFMLP, self).__init__()
        self.block = block
        self.expand_ratio = expand_ratio
        self.mlp_ratio = mlp_ratio
        self.use_dw = use_dw
        self.drop_path_rate = drop_path_rate

        if isinstance(dims, int):
            dims = [dims//2, dims, dims*2, dims*4, dims*8]
        else:
            dims = [dims[0]//2] + dims

        self.first_conv = ConvX(3, dims[0], 1, 3, 2, use_act=True)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]

        self.layer1 = self._make_layers(dims[0], dims[1], layers[0], stride=2, drop_path=dpr[:layers[0]])
        self.layer2 = self._make_layers(dims[1], dims[2], layers[1], stride=2, drop_path=dpr[layers[0]:sum(layers[:2])])
        self.layer3 = self._make_layers(dims[2], dims[3], layers[2], stride=2, drop_path=dpr[sum(layers[:2]):sum(layers[:3])])
        self.layer4 = self._make_layers(dims[3], dims[4], layers[3], stride=2, drop_path=dpr[sum(layers[:3]):sum(layers[:4])])

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = MlpHead(dims[4], num_classes)

        self.init_params(self)

    def _make_layers(self, inputs, outputs, num_block, stride, drop_path):
        layers = [self.block(inputs, outputs, stride, self.expand_ratio, self.mlp_ratio, self.use_dw, drop_path[0])]

        for i in range(1, num_block):
            layers.append(self.block(outputs, outputs, 1, self.expand_ratio, self.mlp_ratio, self.use_dw, drop_path[i]))
            
        return nn.Sequential(*layers)

    def init_params(self, model):
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out = self.gap(x).flatten(1)
        out = self.classifier(out)
        return out

