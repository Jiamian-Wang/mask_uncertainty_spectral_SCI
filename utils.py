import os
import math
import torch
import logging
import numpy as np
import scipy.io as sio
from ssim_torch import ssim



def previous_epochs(last_train,
                    WARMUP_EPOCH,
                    RECON_INTER,
                    NOISE_INTER):

    if last_train < WARMUP_EPOCH:

        recon_epoch_sum, noise_epoch_sum = 0, 0

    else:

        alter_epoch_sum = last_train - WARMUP_EPOCH
        alter_loops = alter_epoch_sum // (RECON_INTER + NOISE_INTER)
        remain_epoch_sum = alter_epoch_sum % (RECON_INTER + NOISE_INTER)

        print('alter_loops=', alter_loops)
        print('alter_epoch_sum=', alter_epoch_sum)
        print('remain_epoch_sum=', remain_epoch_sum)

        recon_epoch_sum = alter_loops * RECON_INTER + remain_epoch_sum

        if remain_epoch_sum <= RECON_INTER:

            noise_epoch_sum = alter_loops * NOISE_INTER
            print('noise_epoch_sum=',noise_epoch_sum)

        else:

            noise_epoch_sum = alter_loops * NOISE_INTER + remain_epoch_sum - RECON_INTER

    return recon_epoch_sum, noise_epoch_sum



def generate_masks(mask_path,
                   batch_size,
                   data_chl):

    mask = sio.loadmat(mask_path + '/mask.mat')

    mask = mask['mask']

    mask3d = np.tile(mask[:,:,np.newaxis],(1,1,data_chl))
    mask3d = np.transpose(mask3d, [2, 0, 1])
    mask3d = torch.from_numpy(mask3d)

    [nC, H, W] = mask3d.shape
    mask3d_batch = mask3d.expand([batch_size, nC, H, W]).cuda().float()

    return mask3d_batch




def LoadTraining(path, scale=True):

    imgs = []
    scene_list = os.listdir(path)
    scene_list.sort()

    print('training sences:', len(scene_list))


    for i in range(len(scene_list)):

        print('start for=',i)
        scene_path = path + scene_list[i]

        if 'mat' not in scene_path:

            continue

        img_dict = sio.loadmat(scene_path)

        if "img_expand" in img_dict:

            img = img_dict['img_expand']/65536.

        elif "img" in img_dict:

            img = img_dict['img']/65536.

        img = img.astype(np.float32)
        imgs.append(img)

        print('Sence {} is loaded. {}'.format(i, scene_list[i]))

    return imgs

def LoadTest_real(path_test):

    scene_list = os.listdir(path_test)
    scene_list.sort()
    test_data = np.zeros((len(scene_list), 660, 714, 1))

    for i in range(len(scene_list)):

        scene_path = path_test + scene_list[i]
        img_dict = np.load(scene_path, allow_pickle=True).item()
        img = img_dict['meas_real']
        test_data[i,:,:,:] = img[:,:,np.newaxis]
        print(i, img.shape, img.max(), img.min())

    test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))

    return test_data

def LoadTest_24chl(path_test):

    test_data = sio.loadmat(path_test)['data']
    test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
    data_chl  = test_data.shape[1]

    return test_data, data_chl

def LoadTest_28chl(path_test, patch_size):

    scene_list = os.listdir(path_test)
    scene_list.sort()
    test_data = np.zeros((len(scene_list), patch_size, patch_size, 28))

    for i in range(len(scene_list)):

        scene_path = path_test + scene_list[i]
        img = sio.loadmat(scene_path)['img']
        test_data[i,:,:,:] = img
        print(i, img.shape, img.max(), img.min())

    test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
    data_chl  = test_data.shape[1]

    return test_data, data_chl


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


def time2file_name(time):

    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second

    return time_filename


def shuffle_crop(train_data, batch_size, patch_size, data_chl):
    
    index = np.random.choice(np.arange(len(train_data)), batch_size)
    processed_data = np.zeros((batch_size, patch_size, patch_size, data_chl), dtype=np.float32)
    
    for i in range(batch_size):

        h, w, _ = train_data[index[i]].shape
        x_index = np.random.randint(0, h - patch_size)
        y_index = np.random.randint(0, w - patch_size)
        processed_data[i, :, :, :] = train_data[index[i]][x_index:x_index + patch_size, y_index:y_index + patch_size, :]

    gt_batch = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2)))

    return gt_batch



def shuffle_crop_mask(mask3d_batch_train, batch_size, patch_size, data_chl):

    mask3d_batch_train = mask3d_batch_train.cpu().float()
    processed_data = np.zeros((batch_size, data_chl, patch_size, patch_size), dtype=np.float32)

    for i in range(batch_size):

        _, h, w = mask3d_batch_train[i].shape
        x_index = np.random.randint(0, h - patch_size)
        y_index = np.random.randint(0, w - patch_size)
        processed_data[i, :, :, :] = mask3d_batch_train[i][:,x_index:x_index + patch_size, y_index:y_index + patch_size]

    gt_batch = torch.from_numpy(processed_data)

    return gt_batch


def gen_meas(data_batch, mask3d_batch, model_type):

    nC = data_batch.shape[1]
    temp = shift(mask3d_batch*data_batch, 2)
    meas = torch.sum(temp, 1)/nC*2
    y_temp = shift_back(meas, data_chl=nC)

    if model_type == 'single_inp':

        PhiTy = torch.mul(y_temp, mask3d_batch)
        return PhiTy

    elif model_type == 'meas&mask':

        return y_temp

    else:

        print('Invalid model_type, should be either [single_inp] or [meas&mask]')
        raise ValueError

def gen_meas_test(data_batch, mask3d_batch, model_type):

    [batch_size, nC, H, W] = data_batch.shape
    mask3d_batch = (mask3d_batch[0,:,:,:]).expand([batch_size, nC, H, W]).cuda().float()
    temp = shift(mask3d_batch*data_batch, 2)
    meas = torch.sum(temp, 1)/nC*2
    y_temp = shift_back(meas, data_chl=nC)

    if model_type == 'single_inp':

        PhiTy = torch.mul(y_temp, mask3d_batch)

        return PhiTy, mask3d_batch

    elif model_type == 'meas&mask':

        return y_temp, mask3d_batch

    else:

        print('Invalid model_type, should be either [single_inp] or [meas&mask]')
        raise ValueError


def shift(inputs, step=2):

    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col+(nC-1)*step).cuda().float()

    for i in range(nC):

        output[:,i,:,step*i:step*i+col] = inputs[:,i,:,:]

    return output

def shift_back(inputs,data_chl,step=2):

    [bs, row, col] = inputs.shape
    nC = data_chl
    output = torch.zeros(bs, nC, row, col-(nC-1)*step).cuda().float()

    for i in range(nC):

        output[:,i,:,:] = inputs[:,:,step*i:step*i+col-(nC-1)*step]

    return output

def gen_log(model_path):

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    
    log_file = model_path + '/log.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO) 
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
