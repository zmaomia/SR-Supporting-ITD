import torch


class HybridLoss(torch.nn.Module):
    def __init__(self, lamd=1e-1, spatial_tv=False, spectral_tv=False):
        super(HybridLoss, self).__init__()
        self.lamd = lamd
        self.use_spatial_TV = spatial_tv
        self.use_spectral_TV = spectral_tv
        self.fidelity = torch.nn.L1Loss()
        self.spatial = TVLoss(weight=1e-3)
        self.spectral = TVLossSpectral(weight=1e-3)

    def forward(self, y, gt):
        loss = self.fidelity(y, gt)
        spatial_TV = 0.0
        spectral_TV = 0.0
        if self.use_spatial_TV:
            spatial_TV = self.spatial(y)
        if self.use_spectral_TV:
            spectral_TV = self.spectral(y)
        total_loss = loss + spatial_TV + spectral_TV
        return total_loss


class reconstruction_SADloss(torch.nn.Module):
    def __init__(self):
        super(reconstruction_SADloss, self).__init__()

    def forward(self, x, y):
        abundance_loss = torch.acos(torch.cosine_similarity(x, y, dim=1))
        abundance_loss = torch.mean(abundance_loss)
        return abundance_loss


class CharbonnierLoss(torch.nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


class AbunConLoss(torch.nn.Module):
    def __init__(self):
        super(AbunConLoss, self).__init__()
        self.loss = CharbonnierLoss()
        k = nn.Paremeter(torch.ones(5, 1, 4, 4) / 256)
        self.convkernel = k.cuda()

    def abunconv(self, abundance):
        return torch.nn.functional.conv2d(abundance,
                                          self.convkernel,
                                          bias=None,
                                          stride=4,
                                          padding=0,
                                          groups=5)

    def forward(self, HRabundance, LRabundance):
        loss = self.loss(self.abunconv(HRabundance), LRabundance)
        return loss





# from https://github.com/jxgu1016/Total_Variation_Loss.pytorch with slight modifications
class TVLoss(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        # h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :h_x - 1, :]).sum()
        # w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x - 1]).sum()
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * (h_tv / count_h +
                                     w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class EdgeLoss(torch.nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(5, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()
        self.downsampling44 = torch.nn.AvgPool2d(4, 4, ceil_mode=True)

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = torch.nn.functional.pad(img,
                                      (kw // 2, kh // 2, kw // 2, kh // 2),
                                      mode='replicate')
        return torch.nn.functional.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)  # filter
        down = filtered[:, :, ::2, ::2]  # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4  # upsample
        filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        x = self.downsampling44(x)
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


class TVLossSpectral(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(TVLossSpectral, self).__init__()
        self.TVLoss_weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        c_x = x.size()[1]
        count_c = self._tensor_size(x[:, 1:, :, :])
        # c_tv = torch.abs((x[:, 1:, :, :] - x[:, :c_x - 1, :, :])).sum()
        c_tv = torch.pow((x[:, 1:, :, :] - x[:, :c_x - 1, :, :]), 2).sum()
        return self.TVLoss_weight * 2 * (c_tv / count_c) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]
