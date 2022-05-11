import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms as T
import cv2
import numpy as np
from torch.autograd import Variable
import torch.nn as nn

def load_image(path, resize, device):
    if not '.' in path:
        path = path + '.png'
    image = Image.open(path, mode='r')
    return process_image(image, resize, device)


def process_image(image, resize, device):
    image = T.ToTensor()(image)
    shape = torch.tensor(image.shape)[1:]
    scale = resize / torch.max(shape)
    shape = (shape * scale).type(torch.int32).tolist()
    image = T.Resize(shape)(image)
    return image.to(device)


def show_image(img):
    img = img.cpu().clone()
    plt.imshow(T.functional.to_pil_image(img))
    plt.show()

def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow

    SOURCE: https://github.com/NVlabs/PWC-Net/blob/master/PyTorch/models/PWCNet.py#L139
    PWC-Net is under Creative Commons license.
    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = Variable(grid) + flo

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = nn.functional.grid_sample(x, vgrid, align_corners=False)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid, align_corners=False)
    
    mask[mask<0.9999] = 0
    mask[mask>0] = 1

    return output * mask


def warp_img(prev_img_stylized, prev_img, curr_img, device):
    """
    Returns an image warped from prev_img_stylized, 
    using the flow from prev_img to img.
    """
    
    # Compute optical flow from prev_img to curr_im (unstylized)
    prev_img = (255 * prev_img.detach().squeeze(0)).cpu().numpy().transpose(1, 2, 0)
    curr_img = (255 * curr_img.detach().squeeze(0)).cpu().numpy().transpose(1, 2, 0)
    im1 = np.float64(cv2.cvtColor(prev_img, cv2.COLOR_RGB2GRAY))
    im2 = np.float64(cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY))
    flow = cv2.calcOpticalFlowFarneback(im1, im2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow = torch.from_numpy(flow.transpose(2,0,1)).unsqueeze(0).float().to(device)
    
    return warp(prev_img_stylized, -flow)