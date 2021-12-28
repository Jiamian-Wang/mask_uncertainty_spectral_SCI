import time
from utils import *
import torch.nn as nn
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

if not torch.cuda.is_available():
    raise Exception('NO GPU!')

mask_path_real256 = './Data'
mask_path_real660 = './Data/real_mask'
test_path = "./Data/testing/"

batch_size = 1
last_train = 661   # 661 for many-to-many, 662 for one-to-X
trial_num = 100
patch_size = 256   # for simulation data

mode = 'many_to_many' # could be 'one_to_one' 'one_to_many' 'many_to_many'
print('\n Testing mode = {:s}\n'.format(mode))

mask3d_batch_randomcrop = generate_masks(mask_path_real660, batch_size)
mask3d_batch_real256 = generate_masks(mask_path_real256, batch_size)


model_train = torch.load('./' + mode + '/model_structure.pth')
print('load model structure from', type(model_train))

if last_train != 0:
    print('load pre-trained params.')
    ckpt_path = './' + mode + '/model_epoch_{}.pth'.format(last_train)
    ckpt = torch.load(ckpt_path)
    model_train.load_state_dict(ckpt['model_train'])

test_set = LoadTest(test_path, patch_size)

def test(test_set,
         patch_size,
         mask_mode,
         use_mask = mask3d_batch_real256):
    psnr_list, ssim_list = [], []

    psnr_htrial_wscene, ssim_htrial_wscene = [], []

    test_gt = test_set.cuda().float()
    if mask_mode == 'real256':
        y, mask3d_test = gen_meas_test(test_gt, use_mask)
        model_train.eval()
        begin = time.time()
        with torch.no_grad():
            model_out = model_train(y, mask3d_test)[0]
        end = time.time()
        for k in range(test_gt.shape[0]):
            psnr_val = torch_psnr(model_out[k, :, :, :], test_gt[k, :, :, :])
            psnr_list.append(psnr_val.detach().cpu().numpy())
            ssim_val = torch_ssim(model_out[k, :, :, :], test_gt[k, :, :, :])
            ssim_list.append(ssim_val.detach().cpu().numpy())
        pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
        truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
        psnr_mean = np.mean(np.asarray(psnr_list))
        ssim_mean = np.mean(np.asarray(ssim_list))
        print('===>testing psnr = {:.2f}, ssim = {:.5f}, time: {:.2f}'.format(psnr_mean, ssim_mean, (end - begin)))

        return pred, truth, psnr_list, ssim_list, psnr_mean, ssim_mean
    elif mask_mode == 'randomcrop':
        psnr_trials = []
        ssim_trials = []

        for trial in range(trial_num):

            psnr_ls_eachtrial = []
            ssim_ls_eachtrial = []

            mask3d_test = shuffle_crop_mask(use_mask, batch_size, patch_size)
            mask3d_test = Variable(mask3d_test).cuda().float()
            y, mask3d_test = gen_meas_test(test_gt, mask3d_test)
            model_train.eval()
            begin = time.time()
            with torch.no_grad():
                model_out = model_train(y, mask3d_test)[0]
            end = time.time()
            for k in range(test_gt.shape[0]):
                psnr_val = torch_psnr(model_out[k,:,:,:], test_gt[k,:,:,:])
                psnr_list.append(psnr_val.detach().cpu().numpy())

                ssim_val = torch_ssim(model_out[k, :, :, :], test_gt[k, :, :, :])
                ssim_list.append(ssim_val.detach().cpu().numpy())

                psnr_ls_eachtrial.append(psnr_val.detach().cpu().numpy())
                ssim_ls_eachtrial.append(ssim_val.detach().cpu().numpy())

            pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
            truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)

            psnr_htrial_wscene.append(psnr_ls_eachtrial)
            ssim_htrial_wscene.append(ssim_ls_eachtrial)

            psnr_mean = np.mean(np.asarray(psnr_list))
            ssim_mean = np.mean(np.asarray(ssim_list))
            psnr_trials.append(psnr_mean)
            ssim_trials.append(ssim_mean)
        psnr_ave = np.mean(psnr_trials)
        psnr_std = np.std(psnr_trials)
        ssim_ave = np.mean(ssim_trials)
        ssim_std = np.std(ssim_trials)
        print('\n===>testing psnr = {:.2f}/{:.5f}, ssim = {:.5f}/{:.5f}, time: {:.2f}'.format(psnr_ave, psnr_std, ssim_ave, ssim_std, (end - begin)), '\n')

        return pred, truth, psnr_list, ssim_list, psnr_ave, ssim_ave, np.array(psnr_htrial_wscene), np.array(ssim_htrial_wscene)
    else:
        print('Invalid mask_mode, should be either [real256] or [randomcrop]')
        raise ValueError


def main():

    if   mode == 'many_to_many':
        pred, truth, psnr_all, ssim_all, psnr_mean, ssim_mean, psnr_hw, ssim_hw = test(test_set,
                                                                                       patch_size,
                                                                                       mask_mode='randomcrop',
                                                                                       use_mask=mask3d_batch_randomcrop
                                                                                       )
        allscene_psnr_mean = np.mean(psnr_hw, axis=0)
        print('psnr mean of 10 scenes are: ', allscene_psnr_mean)

        allscene_psnr_std = np.std(psnr_hw, axis=0)
        print('psnr std of 10 scenes are: ', allscene_psnr_std)

        allscene_ssim_mean = np.mean(ssim_hw, axis=0)
        print('ssim mean of 10 scenes are: ', allscene_ssim_mean)

        allscene_ssim_std = np.std(ssim_hw, axis=0)
        print('ssim std of 10 scenes are: ', allscene_ssim_std)

    elif mode == 'one_to_one':
        pred, truth, psnr_all, ssim_all, psnr_mean, ssim_mean = test(test_set,
                                                                     patch_size,
                                                                     mask_mode='real256',
                                                                     use_mask=mask3d_batch_real256
                                                                     )
    elif mode == 'one_to_many':
        pred, truth, psnr_all, ssim_all, psnr_mean, ssim_mean, psnr_hw, ssim_hw = test(test_set,
                                                                                       patch_size,
                                                                                       mask_mode='randomcrop',
                                                                                       use_mask=mask3d_batch_randomcrop
                                                                                       )

        allscene_psnr_mean = np.mean(psnr_hw, axis=0)
        print('psnr mean of 10 scenes are: ', allscene_psnr_mean)

        allscene_psnr_std = np.std(psnr_hw, axis=0)
        print('psnr std of 10 scenes are: ', allscene_psnr_std)

        allscene_ssim_mean = np.mean(ssim_hw, axis=0)
        print('ssim mean of 10 scenes are: ', allscene_ssim_mean)

        allscene_ssim_std = np.std(ssim_hw, axis=0)
        print('ssim std of 10 scenes are: ', allscene_ssim_std)

    else:
        print('INVALID [mode]!!')
        raise ValueError


if __name__ == '__main__':
    main()
    

