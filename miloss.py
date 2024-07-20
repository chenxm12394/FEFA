import math
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from packaging import version

def gaussian_kernel_1d(sigma):
    kernel_size = int(2*math.ceil(sigma*2) + 1)
    x = torch.linspace(-(kernel_size-1)//2, (kernel_size-1)//2, kernel_size)
    kernel = 1.0/(sigma*math.sqrt(2*math.pi))*torch.exp(-(x**2)/(2*sigma**2))
    kernel = kernel/torch.sum(kernel)
    return kernel

def gaussian_kernel_2d(sigma):
    y_1 = gaussian_kernel_1d(sigma[0])
    y_2 = gaussian_kernel_1d(sigma[1])
    kernel = torch.tensordot(y_1, y_2, 0)
    kernel = kernel / torch.sum(kernel)
    return kernel

def gaussian_smooth(img, sigma):
    kernel = gaussian_kernel_2d((sigma, sigma))[None, None, :, :].to(img)
    padding = kernel.shape[-1]//2
    img = torch.nn.functional.conv2d(img, kernel, padding=padding)
    return img

def compute_marginal_entropy(values, bins, sigma):
    normalizer_1d = np.sqrt(2.0 * np.pi) * sigma
    sigma = 2*sigma**2
    p = torch.exp(-((values - bins).pow(2).div(sigma))).div(normalizer_1d)
    p_n = p.mean(dim=1)
    p_n = p_n/(torch.sum(p_n) + 1e-10)
    return -(p_n * torch.log(p_n + 1e-10)).sum(), p



def _mi_loss(I, J, bins, sigma):
    # compute marjinal entropy
    ent_I, p_I = compute_marginal_entropy(I.view(-1), bins, sigma)
    ent_J, p_J = compute_marginal_entropy(J.view(-1), bins, sigma)
    # compute joint entropy
    normalizer_2d = 2.0 * np.pi*sigma**2
    p_joint = torch.mm(p_I, p_J.transpose(0, 1)).div(normalizer_2d)
    p_joint = p_joint / (torch.sum(p_joint) + 1e-10)
    ent_joint = -(p_joint * torch.log(p_joint + 1e-10)).sum()

    return -(ent_I + ent_J - ent_joint)


def mi_loss(I, J, bins=64 ,sigma=1.0/64, minVal=0, maxVal=1):
    #if sigma > 1:
    #    kernel = gaussian_kernel_2d((sigma, sigma))[None, None, :, :].to(I)
    #    padding = kernel.shape[-1]//2
    #    I = torch.nn.functional.conv2d(I, kernel, padding=padding)
    #    J = torch.nn.functional.conv2d(J, kernel, padding=padding)
    bins = torch.linspace(minVal, maxVal, bins).to(I).unsqueeze(1)
    neg_mi =[_mi_loss(I, J, bins, sigma) for I, J in zip(I, J)]
    return sum(neg_mi)/len(neg_mi)

def ms_mi_loss(I, J, bins=64, sigma=1.0/64, ms=3, smooth=3, minVal=0, maxVal=1):
    smooth_fn = lambda x: torch.nn.functional.avg_pool2d( \
            gaussian_smooth(x, smooth), kernel_size = 2, stride=2)
    loss = mi_loss(I, J, bins=bins, sigma=sigma, minVal=minVal, maxVal=maxVal)
    for _ in range(ms - 1):
        I, J = map(smooth_fn, (I, J))
        loss = loss + mi_loss(I, J, \
                bins=bins, sigma=sigma, minVal=minVal, maxVal=maxVal)
    return loss / ms

class gradient_loss(torch.nn.Module):
    def __init__(self):
        super(gradient_loss, self).__init__()

    def forward(self, s, penalty='l2'):
        dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
        dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])
        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
        d = torch.mean(dx) + torch.mean(dy)
        return d / 2.0


class LossFunction_Dense(torch.nn.Module):
    def __init__(self):
        super(LossFunction_Dense, self).__init__()
        self.gradient_loss = gradient_loss()
        # self.feat_loss = VGGLoss()

    def forward(self, y, y_f, tgt, src, flow): # tgt: torch.Size([16, 1, 224, 224])

        hyper_ncc = 1
        hyper_grad = 10
        hyper_feat = 1
       
        ncc_1 = torch.nn.functional.l1_loss(tgt, y)
        ncc_2 = torch.nn.functional.l1_loss(src, y_f)
        ncc = ncc_1 + 0.2*ncc_2

        # TODO: feature loss
        # feat_1 = self.feat_loss(y, tgt)
        # feat_2 = self.feat_loss(y_f, src)
        # feat = feat_1 + 0.2*feat_2

        # TODO: gradient loss
        grad = self.gradient_loss(flow)

        # TODO: multi-scale loss
        # multi_1 = self.multi_loss(src, tgt, flow1, flow2, hyper_3, hyper_4)
        # multi_2 = self.multi_loss(tgt, src, -flow1, -flow2, hyper_3, hyper_4)
        # multi = multi_1 + 0.2*multi_2

        # TODO: edge loss
        # edge_1 = self.edge_loss(y, tgt)
        # edge_2 = self.edge_loss(y_f, src)
        # edge = edge_1 + edge_2

        # TODO: total loss
        # loss = multi + hyper_ncc * ncc + hyper_grad * grad + hyper_feat * feat
        # return loss, multi, ncc, grad
        # loss = hyper_grad * grad + hyper_feat * feat
        # loss = hyper_feat * feat + hyper_grad * grad
        # return loss, feat, ncc, grad
        return ncc, grad

'''
def compute_joint_prob(x, y, bins):
    p = torch.exp(-(x-bins).pow(2).unsqueeze(1)-(y-bins).pow(2).unsqueeze(0))
    p_n = p.mean(dim=2)
    p_n = p_n/(torch.sum(p_n) + 1e-10)
    return p_n

def _mi_loss1(I, J, bins):
    # in fact _mi_loss1 and _mi_loss works in the same way,
    # with minor different cased by numerical error
    Pxy = compute_joint_prob(I.view(-1), J.view(-1), bins)
    Px = Pxy.sum(axis=1)
    Py = Pxy.sum(axis=0)
    PxPy = Px[..., None]*Py[None, ...]
    mi = Pxy*(torch.log(Pxy+1e-10) - torch.log(PxPy+1e-10))
    return -mi.sum()
'''

class PatchNCELoss(torch.nn.Module):
    def __init__(self, T = 0.07):
        super().__init__()
        
        self.T = T
        
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.bool
    def forward(self, feat_q, feat_k,weight=None):
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        batch_dim_for_bmm = 1

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))
        if weight is not None:
            l_neg_curbatch *= weight

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.T
        # print('out', out)

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X) # torch.Size([1, 64, 256, 256])
        h_relu2 = self.slice2(h_relu1) # torch.Size([1, 128, 128, 128])
        h_relu3 = self.slice3(h_relu2) # torch.Size([1, 256, 64, 64])
        h_relu4 = self.slice4(h_relu3) # torch.Size([1, 512, 32, 32])
        h_relu5 = self.slice5(h_relu4) # torch.Size([1, 512, 16, 16])
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(torch.nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        if torch.cuda.is_available():
            self.vgg.cuda()
        self.vgg.eval()
        set_requires_grad(self.vgg, False)
        self.L1Loss = torch.nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 , 1.0]

    def forward(self, x, y):
        contentloss = 0
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        x_vgg = self.vgg(x)
        with torch.no_grad():
            y_vgg = self.vgg(y)

        contentloss += self.L1Loss(x_vgg[3], y_vgg[3].detach())

        return contentloss

class CharbonnierLoss(torch.nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(torch.nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss
    

if __name__ == '__main__':
    #data = np.random.multivariate_normal( \
    #        mean=[0,0], cov=[[1,0.8],[0.8,1]], size=1000)
    #x, y = data[:,0], data[:,1]
    # noise = 0.1
    # x = np.random.random(512*512)*(1-noise)
    # y = x + np.random.random(512*512)*noise


    # from sklearn.metrics import mutual_info_score
    # def calc_MI(x, y, bins):
    #     c_xy = np.histogram2d(x, y, bins, range=((0,1),(0,1)))[0]
    #     mi = mutual_info_score(None, None, contingency=c_xy)
    #     return mi
    # print(calc_MI(x, y, 64))

    # #from scipy.stats import chi2_contingency
    # #def calc_MI(x, y, bins):
    # #    c_xy = np.histogram2d(x, y, bins)[0]
    # #    g, p, dof, expected = chi2_contingency(c_xy, lambda_="log-likelihood")
    # #    mi = 0.5 * g / c_xy.sum()
    # #    return mi
    # #print(calc_MI(x, y, 60))

    # from scipy.special import xlogy
    # def calc_MI(x, y, bins):
    #     Pxy = np.histogram2d(x, y, bins, range=((0,1),(0,1)))[0]
    #     Pxy = Pxy/Pxy.sum()
    #     Px = Pxy.sum(axis=1)
    #     Py = Pxy.sum(axis=0)
    #     PxPy = Px[..., None]*Py[None, ...]
    #     #mi = Pxy * np.log(Pxy/(PxPy+1e-6))
    #     mi = xlogy(Pxy, Pxy) - xlogy(Pxy, PxPy)
    #     return mi.sum()
    # print(calc_MI(x, y, 64))

    # print(-mi_loss(torch.Tensor([x]), torch.Tensor([y]), bins=64 ,sigma=3, minVal=0, maxVal=1))
    a = torch.rand((4, 1, 320, 320))
    b = torch.rand((4, 1, 320, 320))
    print(ncc_loss(a, b))


