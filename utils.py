import os
import math
import torch 
import numpy as np
import scipy.io as sio
from ssim_torch import ssim

def generate_masks(mask_path,
                   batch_size):

    mask = sio.loadmat(mask_path + '/mask.mat')
    mask = mask['mask']

    mask3d = np.tile(mask[:,:,np.newaxis],(1,1,28))
    mask3d = np.transpose(mask3d, [2, 0, 1])
    mask3d = torch.from_numpy(mask3d)
    [nC, H, W] = mask3d.shape
    mask3d_batch = mask3d.expand([batch_size, nC, H, W]).cuda().float()
    return mask3d_batch

def LoadTest(path_test, patch_size):
    scene_list = os.listdir(path_test)
    scene_list.sort()
    test_data = np.zeros((len(scene_list), patch_size, patch_size, 28))
    for i in range(len(scene_list)):
        scene_path = path_test + scene_list[i]
        img_dict = np.load(scene_path, allow_pickle=True).item()
        img = img_dict['img']
        test_data[i,:,:,:] = img
        print(i, img.shape, img.max(), img.min())
    test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
    return test_data

def psnr(img1, img2):
    psnr_list = []
    for i in range(img1.shape[0]):
        total_psnr = 0
        PIXEL_MAX = img2[i,:,:,:].max()
        for ch in range(28):
            mse = np.mean((img1[i,:,:,ch] - img2[i,:,:,ch])**2)
            total_psnr += 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
        psnr_list.append(total_psnr/img1.shape[3])
    return psnr_list

def torch_psnr(img, ref):
    nC = img.shape[0]
    pixel_max = torch.max(ref)
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i,:,:] - ref[i,:,:]) ** 2)
        psnr += 20 * torch.log10(pixel_max / torch.sqrt(mse))
    return psnr/nC

def torch_ssim(img, ref):
    return ssim(torch.unsqueeze(img,0), torch.unsqueeze(ref,0))

def shuffle_crop(train_data, batch_size, patch_size):
    
    index = np.random.choice(np.arange(len(train_data)), batch_size)
    processed_data = np.zeros((batch_size, patch_size, patch_size, 28), dtype=np.float32)
    
    for i in range(batch_size):
        h, w, _ = train_data[index[i]].shape
        x_index = np.random.randint(0, h - patch_size)
        y_index = np.random.randint(0, w - patch_size)
        processed_data[i, :, :, :] = train_data[index[i]][x_index:x_index + patch_size, y_index:y_index + patch_size, :]  # change
    gt_batch = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2)))
    return gt_batch

def shuffle_crop_mask(mask3d_batch_train, batch_size, patch_size):
    mask3d_batch_train = mask3d_batch_train.cpu().float()
    processed_data = np.zeros((batch_size, 28, patch_size, patch_size), dtype=np.float32)

    for i in range(batch_size):
        _, h, w = mask3d_batch_train[i].shape
        x_index = np.random.randint(0, h - patch_size)
        y_index = np.random.randint(0, w - patch_size)
        processed_data[i, :, :, :] = mask3d_batch_train[i][:,x_index:x_index + patch_size, y_index:y_index + patch_size]  # change
    gt_batch = torch.from_numpy(processed_data)
    return gt_batch

def gen_meas_test(data_batch, mask3d_batch):
    [batch_size, nC, H, W] = data_batch.shape
    mask3d_batch = (mask3d_batch[0,:,:,:]).expand([batch_size, nC, H, W]).cuda().float()
    temp = shift(mask3d_batch*data_batch, 2)
    meas = torch.sum(temp, 1)/nC*2
    y_temp = shift_back(meas)
    return y_temp, mask3d_batch

def shift(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col+(nC-1)*step).cuda().float()
    for i in range(nC):
        output[:,i,:,step*i:step*i+col] = inputs[:,i,:,:]
    return output

def shift_back(inputs,step=2):
    [bs, row, col] = inputs.shape
    nC = 28
    output = torch.zeros(bs, nC, row, col-(nC-1)*step).cuda().float()
    for i in range(nC):
        output[:,i,:,:] = inputs[:,:,step*i:step*i+col-(nC-1)*step]
    return output
