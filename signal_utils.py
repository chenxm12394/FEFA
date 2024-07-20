import torch
import torch.fft
import numpy as np
import matplotlib.pyplot as plt

def fft2(x):
    assert len(x.shape) == 4
    x = torch.fft.fft2(x, norm='ortho')
    return x

def ifft2(x):
    assert len(x.shape) == 4
    x = torch.fft.ifft2(x, norm='ortho')
    return x

def fftshift2(x):
    assert len(x.shape) == 4
    x = torch.roll(x, (x.shape[-2]//2, x.shape[-1]//2), dims=(-2, -1))
    return x

def ifftshift2(x):
    assert len(x.shape) == 4
    x = torch.roll(x, ((x.shape[-2]+1)//2, (x.shape[-1]+1)//2), dims=(-2, -1))
    return x

def rss(x):
    assert len(x.shape) == 4
    return torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)
    #if torch.is_complex(x):
    #    return (x.real**2 + x.imag**2).sum(dim=1, keepdim=True).sqrt()
    #else:
    #    return (x**2).sum(dim=1, keepdim=True)**0.5

def extract_amp_spectrum(trg_img):

    fft_trg = fft2(trg_img)
    amp_target, pha_trg = torch.abs(fft_trg), torch.angle(fft_trg)

    return amp_target

def amp_spectrum_swap( amp_local, amp_target, L=0.1 , ratio=0):
    
    h, w = amp_local.shape[-2:]
    print('hw', h, w)
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    amp_local[..., h1:h2,w1:w2] = amp_local[..., h1:h2,w1:w2] * ratio + amp_target[..., h1:h2,w1:w2] * (1- ratio)
    return amp_local

def freq_space_interpolation( traget_img, aux_img, L=0 , ratio=0):
    print(traget_img.shape, aux_img.shape)
    # get fft of local sample
    fft_trg = fftshift2(fft2(traget_img))
    amp_target, pha_trg = torch.abs(fft_trg), torch.angle(fft_trg)

    fft_aux = fftshift2(fft2(aux_img))
    amp_aux, pha_aux = torch.abs(fft_aux), torch.angle(fft_aux)

    # swap the amplitude part of local image with target amplitude spectrum
    amp_aux_ = amp_spectrum_swap( amp_aux, amp_target, L=0.04 , ratio=0.5)

    # get transformed image via inverse fft
    fft_aux_ = amp_aux_ * torch.exp( 1j * pha_aux )
    aux_in_trg = ifftshift2(fft_aux_)
    aux_in_trg = ifft2(aux_in_trg)
    aux_in_trg = torch.real(aux_in_trg)
    plt.imsave('batch_1.png', aux_in_trg[0][0].cpu().numpy(), cmap='gray')

    return aux_in_trg