import time
import argparse
from utils import *
import torch.nn as nn
from network.ST import ST_MODEL
from network.GST import GST_MODEL
from torch.autograd import Variable


parser = argparse.ArgumentParser()
# GPU setting #################################################
parser.add_argument('--device',                   default='0',              help='CUDA ID(s)')
# directories #################################################
parser.add_argument('--mask_path_real256',        default="./Data",           help="256x256 mask dir. used for single mask testing")
parser.add_argument('--mask_path_real660',        default="./Data/real_mask", help="660x660 mask dir. used for random 256x256 mask cropping")
parser.add_argument('--test_path',                default="./Data/testing/28chl/",  help="testing .mat file dir")
parser.add_argument('--test_data_type',           default="28chl",            help="both 24-chl & 28-chl test data are enabled", choices=['24chl', '28chl'])
# optimization #################################################
parser.add_argument("--model_type",               default='GST',  help="GST network or simplified version, ST network", choices=['ST', 'GST'])
parser.add_argument("--batch_size",    type=int,  default=1,      help="batch size for testing, default=1")
parser.add_argument("--last_train",    type=int,  default=661,    help='checkpoint epoch number.')
parser.add_argument('--trial_num',     type=int,  default=100,    help='trial numbers for random mask testing')
parser.add_argument("--patch_size",    type=int,  default=256,    help='training/testing data spatial size, [256] for simulation data')
parser.add_argument("--noise_mean",    type=float,default=0.006,  help='mask noise prior, i.e., Gaussian mean value')
parser.add_argument("--noise_std",     type=float,default=0.006,  help='mask noise prior, i.e., Gaussian std value')
parser.add_argument('--params_init',              default="xavier_uniform",   help="learnable param initializer for GST net", choices=['xavier_uniform', 'uniform', 'normal'])
parser.add_argument("--inter_channels",type=int,  default=28,     help='embedding channel in GST net')
parser.add_argument("--spatial_scale", type=int,  default=4,      help='down-scale ratio in GST net')
parser.add_argument('--noise_act',                default="softplus",         help="the last activation function of GST net")
parser.add_argument('--mode',                     default="many_to_many",     help="traditional/miscalibration scenarios", choices=['many_to_many', 'one_to_one', 'one_to_many'])
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')


# customized activation function could be appended here
noise_act_dict = {'softplus': nn.Softplus(),}


if args.test_data_type == '28chl':
    test_set, data_chl = LoadTest_28chl(args.test_path, args.patch_size)
elif args.test_data_type =='24chl':
    test_set, data_chl = LoadTest_24chl(args.test_path)
else:
    print('ERROR: invalid test data type, only support 24chl & 28chl data')
    raise ValueError


mask3d_batch_randomcrop = generate_masks(args.mask_path_real660, args.batch_size, data_chl=data_chl)
mask3d_batch_real256 = generate_masks(args.mask_path_real256, args.batch_size, data_chl=data_chl)

if args.model_type == 'ST':

    model_train = ST_MODEL(in_ch=data_chl,
                           out_ch=data_chl,
                           noise_mean=args.noise_mean,
                           noise_std=args.noise_std,
                           init=args.params_init,
                           noise_act = noise_act_dict[args.noise_act]).cuda()

elif args.model_type == 'GST':

    model_train = GST_MODEL(in_ch=data_chl,
                           out_ch=data_chl,
                           noise_mean=args.noise_mean,
                           noise_std=args.noise_std,
                           init=args.params_init,
                           noise_act=noise_act_dict[args.noise_act],
                           inter_channels=args.inter_channels,
                           spatial_scale=args.spatial_scale).cuda()
else:
    print('ERROR: invalid model type!')
    raise ValueError


assert args.last_train != 0, 'ERROR: last_train must be positive integer!'
ckpt_path = './model/model_epoch_{}.pth'.format(args.last_train)
ckpt = torch.load(ckpt_path)
model_train.load_state_dict(ckpt['model_train'])
print('-----pre-trained params loaded!-----')



def test(test_set,
         patch_size,
         mask_mode,
         trial_num,
         batch_size,
         use_mask = mask3d_batch_real256,
         ):

    psnr_list, ssim_list = [], []

    psnr_htrial_wscene, ssim_htrial_wscene = [], []

    test_gt = test_set.cuda().float()

    if mask_mode == 'real256': # 256x256 real mask do a single-shot testing

        y, mask3d_test = gen_meas_test(test_gt, use_mask, model_type='meas&mask')
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

    elif mask_mode == 'randomcrop': # random crop 256x256 real masks from the 660x660 real mask, for multi-trials testing

        psnr_trials = []
        ssim_trials = []

        for trial in range(trial_num):

            psnr_ls_eachtrial = []
            ssim_ls_eachtrial = []

            mask3d_test = shuffle_crop_mask(use_mask, batch_size, patch_size, data_chl=data_chl)
            mask3d_test = Variable(mask3d_test).cuda().float()
            y, mask3d_test = gen_meas_test(test_gt, mask3d_test, model_type = 'meas&mask')
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

        print('ERROR: Invalid mask_mode, should be either [real256] or [randomcrop]')
        raise ValueError



def main():
    if args.mode == 'many_to_many':

        pred, truth, psnr_all, ssim_all, psnr_mean, ssim_mean, psnr_hw, ssim_hw = test(test_set=test_set,
                                                                                       patch_size=args.patch_size,
                                                                                       trial_num=args.trial_num,
                                                                                       mask_mode='randomcrop',
                                                                                       batch_size=args.batch_size,
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

    elif args.mode == 'one_to_one':

        pred, truth, psnr_all, ssim_all, psnr_mean, ssim_mean = test(test_set=test_set,
                                                                     patch_size=args.patch_size,
                                                                     trial_num=args.trial_num,
                                                                     mask_mode='real256',
                                                                     batch_size=args.batch_size,
                                                                     use_mask=mask3d_batch_real256
                                                                     )

    elif args.mode == 'one_to_many':
        pred, truth, psnr_all, ssim_all, psnr_mean, ssim_mean, psnr_hw, ssim_hw = test(test_set=test_set,
                                                                                       patch_size=args.patch_size,
                                                                                       mask_mode='randomcrop',
                                                                                       trial_num=args.trial_num,
                                                                                       batch_size=args.batch_size,
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

        print('ERROR: INVALID [mode]!')
        raise ValueError


if __name__ == '__main__':

    main()
    

