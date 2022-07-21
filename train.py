
import os
import time
import torch
import datetime
import argparse
import numpy as np
from utils import *
import torch.nn as nn
from tqdm import tqdm, trange
from network.ST import ST_MODEL
from network.GST import GST_MODEL
from torch.autograd import Variable


parser = argparse.ArgumentParser()
# GPU setting #################################################
parser.add_argument('--device',                   default='0',              help='CUDA ID(s)')
# directories #################################################
parser.add_argument('--mask_path_real256',        default="/data/jiamianw/NIPS2021/Data",           help="256x256 mask dir. used for single mask testing")
parser.add_argument('--mask_path_real660',        default="/data/jiamianw/NIPS2021/Data/real_mask", help="660x660 mask dir. used for random 256x256 mask cropping")
parser.add_argument('--test_path',                default="./Data/testing/",  help="testing .mat file dir")
parser.add_argument('--train_path',               default="/data/jiamianw/NIPS2021/Data/training/simu/debug_mat/",  help="train .mat file dir")
parser.add_argument('--val_path',                 default="/data/jiamianw/NIPS2021/Data/validation/fromKAIST_mat/",  help="validation .mat file dir")
parser.add_argument('--model_dir',                default="model_GST",        help="checkpoint dir")
parser.add_argument('--model_save_filename',      default="model_GST",        help="checkpoint folder for resume, set as '' for training from stratch")
parser.add_argument("--psnr_set",                 type=int,  default=10,      help="PSNR value to start save checkpoint")
parser.add_argument("--batch_size",    type=int,  default=4,       help="batch size for testing, default=1")
parser.add_argument("--warmup_lr",     type=float,default=0.0004,  help='initial lr for warm up phase')
parser.add_argument("--recon_lr",      type=float,default=0.0004,  help='initial lr for training phase')
parser.add_argument("--noise_lr",      type=float,default=0.00001, help='initial lr for validation phase')
parser.add_argument("--recon_lr_epoch",type=int,  default=50,      help="lr schedule for training")
parser.add_argument("--noise_lr_epoch",type=int,  default=250,     help="lr schedule for validation")
parser.add_argument("--recon_lr_scale",type=float,default=0.5,     help='initial lr for training phase')
parser.add_argument("--noise_lr_scale",type=float,default=0.5,     help='initial lr for validation phase')
parser.add_argument("--entropy_term",  type=float,default=1.0,     help='entropy term weight')
parser.add_argument("--WARMUP_EPOCH",  type=int,  default=20,      help="epochs for warm up phase")
parser.add_argument("--RECON_INTER",   type=int,  default=3,       help="training epochs in bilevel opt")
parser.add_argument("--NOISE_INTER",   type=int,  default=5,       help="validation epochs in bilevel opt")
parser.add_argument("--stop_criteria", type=int,  default=300,     help="stop training criteria")

parser.add_argument("--epoch_sum_num", type=int,  default=5000,   help="#training samples per epoch")
parser.add_argument("--last_train",    type=int,  default=0,      help='checkpoint epoch number for resume, set as 0 for training from stratch')
parser.add_argument('--trial_num',     type=int,  default=100,    help='trial numbers for random mask testing')
parser.add_argument("--patch_size",    type=int,  default=256,    help='training/testing data spatial size, [256] for simulation data')
parser.add_argument("--noise_patch_size",type=int,default=128,    help='validation/testing data spatial size, [256] for simulation data')

parser.add_argument("--noise_mean",    type=float,default=0.006,  help='mask noise prior, i.e., Gaussian mean value')
parser.add_argument("--noise_std",     type=float,default=0.006,  help='mask noise prior, i.e., Gaussian std value')
parser.add_argument('--params_init',              default="xavier_uniform",   help="learnable param initializer for GST net", choices=['xavier_uniform', 'uniform', 'normal'])
parser.add_argument("--inter_channels",type=int,  default=28,     help='embedding channel in GST net')
parser.add_argument("--data_chl",      type=int,  default=28,     help='24chl or 28chl')

parser.add_argument("--spatial_scale", type=int,  default=4,      help='down-scale ratio in GST net')
parser.add_argument('--noise_act',                default="softplus",         help="the last activation function of GST net")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

batch_num = int(np.floor(args.epoch_sum_num/args.batch_size))

# customized activation function could be appended here
noise_act_dict = {'softplus': nn.Softplus(),}


mask3d_batch_randomcrop = generate_masks(args.mask_path_real660,args.batch_size)
mask3d_batch_real256    = generate_masks(args.mask_path_real256,args.batch_size)

if args.model_type == 'ST':

    model_train = ST_MODEL(in_ch=args.data_chl,
                           out_ch=args.data_chl,
                           noise_mean=args.noise_mean,
                           noise_std=args.noise_std,
                           init=args.params_init,
                           noise_act = noise_act_dict[args.noise_act]).cuda()

elif args.model_type == 'GST':

    model_train = GST_MODEL(in_ch=args.data_chl,
                           out_ch=args.data_chl,
                           noise_mean=args.noise_mean,
                           noise_std=args.noise_std,
                           init=args.params_init,
                           noise_act=noise_act_dict[args.noise_act],
                           inter_channels=args.inter_channels,
                           spatial_scale=args.spatial_scale).cuda()
else:
    print('ERROR: invalid model type!')
    raise ValueError


train_set = LoadTraining(args.train_path, scale=True )
val_set   = LoadTraining(args.val_path,   scale=False)
test_set  = LoadTest_28chl(args.test_path, args.patch_size)


if args.last_train != 0:
    ckpt_path = './'+args.model_dir+'/' + args.model_save_filename + '/model_epoch_{}.pth'.format(args.last_train)
    ckpt = torch.load(ckpt_path)
    try:
        model_train.load_state_dict(ckpt['model_train'])
    except:

        pass

mse = torch.nn.MSELoss().cuda()

class LossIsNaN(Exception):
    pass

def compute_entropy(res_noise):

    return torch.mean(torch.log(res_noise * math.sqrt(2*math.pi*math.e)))



def save(psnr_mean,psnr_max,psnr_set,epoch, model_path,logger):
    '''
    for CKPT, only save information of current epoch, including model checkpoint, metrics, estimations etc.
    '''

    print('epoch %d, better performance, record CKPT!'%epoch)

    if psnr_mean >= psnr_max:

        psnr_max = psnr_mean

        if psnr_mean > psnr_set:

            checkpoint(epoch, model_path, logger)

    return psnr_max



def train(model,
          train_set,
          epoch,
          optimizer,
          batch_num,
          patch_size,
          phase,
          logger,
          entropy_term):

    model.train()

    epoch_loss = 0.0
    epoch_loss_entropy = 0.0
    global_item_num = 0

    begin = time.time()
    print('--epoch%d statred--'%epoch)
    with trange(batch_num) as tepoch:

        for i in tepoch:

            gt_batch = shuffle_crop(train_set, args.batch_size, patch_size)
            gt = Variable(gt_batch).cuda().float()

            if args.mode == 'one_to_many' or args.mode == 'one_to_one':

                if patch_size>128:
                    mask3d_train = mask3d_batch_real256
                else:
                    mask3d_train = shuffle_crop_mask(mask3d_batch_real256, args.batch_size, patch_size)
                    mask3d_train = Variable(mask3d_train).cuda().float()

            elif args.mode == 'many_to_many':

                mask3d_train = shuffle_crop_mask(mask3d_batch_randomcrop, args.batch_size, patch_size)
                mask3d_train = Variable(mask3d_train).cuda().float()

            else:

                print('ERROR: invalid mode type!')
                raise ValueError

            y = gen_meas(gt, mask3d_train, model_type='meas&mask')

            optimizer.zero_grad()

            model_out, sigma = model(y, mask3d_train)

            if phase == 'RECON' or 'WARMUP':
                loss = torch.sqrt(mse(model_out, gt))
            elif phase == 'NOISE':
                loss = torch.sqrt(mse(model_out, gt)) - entropy_term * compute_entropy(sigma)
            else:
                print('ERROR: Invalid phase, should be either [WARMUP] or [RECON] or [NOISE]')
                raise ValueError

            epoch_loss += loss.item() * gt_batch.size(0)


            if np.isnan(float(loss.item())):
                print("Loss is NaN.")
                raise LossIsNaN

            loss.backward()


            if phase=='NOISE':

                tepoch.set_postfix(loss=(torch.sqrt(mse(model_out, gt))- entropy_term * compute_entropy(sigma)).item(),
                                   mse=torch.sqrt(mse(model_out, gt)).item(),
                                   entropy=compute_entropy(sigma).item(),
                                   sigma_ave = sigma.mean().item(),
                                   sigma_min = sigma.min().item(),
                                   sigma_max = sigma.max().item(),
                                   )

                epoch_loss_entropy += compute_entropy(sigma).item()* gt_batch.size(0)

            else:

                tepoch.set_postfix(mse=torch.sqrt(mse(model_out, gt)).item(),
                                   sigma_ave=sigma.mean().item(),
                                   sigma_min=sigma.min().item(),
                                   sigma_max=sigma.max().item()
                                   )

            optimizer.step()

            global_item_num += gt_batch.size(0)

    end = time.time()

    epoch_loss = epoch_loss / global_item_num
    epoch_loss_entropy = epoch_loss_entropy / global_item_num
    print('--epoch%d finished--'%epoch)

    logger.info("===> {:s} Epoch {} Complete: Avg.Loss:{:.6f} Avg.Entropy:{:.6f} time: {:.2f}mins".format(phase,
                                                                                                          epoch,
                                                                                                          epoch_loss,
                                                                                                          epoch_loss_entropy,
                                                                                                          (end - begin)/60.))

    return epoch_loss

def test(test_set,
         epoch,
         logger,
         patch_size,
         mask_mode,
         use_mask = mask3d_batch_real256):

    psnr_list, ssim_list = [], []

    test_gt = test_set.cuda().float()

    if mask_mode == 'real256':

        y, mask3d_test = gen_meas_test(test_gt, use_mask, model_type='meas&mask')

        model_train.eval()

        begin = time.time()
        with torch.no_grad():
            model_out = model_train(y, mask3d_test)[0]
        end = time.time()

        for k in range(test_gt.shape[0]):

            psnr_val = torch_psnr(model_out[k, :, :, :], test_gt[k, :, :, :])
            ssim_val = torch_ssim(model_out[k, :, :, :], test_gt[k, :, :, :])
            psnr_list.append(psnr_val.detach().cpu().numpy())
            ssim_list.append(ssim_val.detach().cpu().numpy())

        psnr_10mean = np.mean(np.asarray(psnr_list))
        ssim_10mean = np.mean(np.asarray(ssim_list))

        logger.info(
            '===> {:s} mask || Epoch {}: testing psnr = {:.3f}, ssim = {:.5f}, time: {:.3f}mins'.format(mask_mode,
                                                                                                        epoch,
                                                                                                        psnr_10mean,
                                                                                                        ssim_10mean,
                                                                                                        (end - begin) / 60.))
        return psnr_10mean, ssim_10mean

    elif mask_mode == 'randomcrop':

        psnr_trials, ssim_trials = [], []

        for trial in range(args.trial_num):

            if patch_size > 128:

                mask3d_test = mask3d_batch_real256

            else:

                mask3d_test = shuffle_crop_mask(mask3d_batch_real256, args.batch_size, patch_size)
                mask3d_test = Variable(mask3d_test).cuda().float()

            y, mask3d_test = gen_meas_test(test_gt, mask3d_test, model_type='meas&mask')

            model_train.eval()

            begin = time.time()
            with torch.no_grad():
                model_out = model_train(y, mask3d_test)[0]
            end = time.time()

            for k in range(test_gt.shape[0]):

                psnr_val = torch_psnr(model_out[k, :, :, :], test_gt[k, :, :, :])
                ssim_val = torch_ssim(model_out[k, :, :, :], test_gt[k, :, :, :])

                psnr_list.append(psnr_val.detach().cpu().numpy())
                ssim_list.append(ssim_val.detach().cpu().numpy())

            psnr_10mean = np.mean(np.asarray(psnr_list))
            ssim_10mean = np.mean(np.asarray(ssim_list))

            psnr_trials.append(psnr_10mean)
            ssim_trials.append(ssim_10mean)

        psnr_ave = np.mean(psnr_trials)
        psnr_std = np.std(psnr_trials)
        ssim_ave = np.mean(ssim_trials)
        ssim_std = np.std(ssim_trials)

        logger.info(
            '===> {:s} mask || {} trials || Epoch {}: testing psnr mean/std= {:.5f}/{:.5f}, ssim mean/std = {:.5f}/{:.5f}'.format(mask_mode,
                                                                                                                        args.trial_num,
                                                                                                                        epoch,
                                                                                                                        psnr_ave,
                                                                                                                        psnr_std,
                                                                                                                        ssim_ave,
                                                                                                                        ssim_std))
        return psnr_ave, ssim_ave, psnr_std, ssim_std

    else:

        print('ERROR: Invalid mask_mode, should be either [real256] or [randomcrop]')
        raise ValueError


    
def checkpoint(epoch, model_path, logger):

    model_out_path = './' + model_path + "/model_epoch_{}.pth".format(epoch)
    torch.save({'model_train':model_train.state_dict()}, model_out_path)
    logger.info("Checkpoint saved to {}".format(model_out_path))
     
def main(warmup_lr,
         recon_lr,
         noise_lr):

    if args.model_save_filename == '':

        date_time = str(datetime.datetime.now())
        date_time = time2file_name(date_time)

    else:

        date_time = args.model_save_filename

    model_path = args.model_dir + '/' + date_time

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    logger = gen_log(model_path)

    psnr_max = 0

    epoch = args.last_train + 1

    recon_epoch_sum, noise_epoch_sum = previous_epochs(args.last_train,
                                                       args.WARMUP_EPOCH,
                                                       args.RECON_INTER,
                                                       args.NOISE_INTER)

    print('Loading || recon_epoch_sum=', recon_epoch_sum, '|| noise_epoch_sum=', noise_epoch_sum)

    Continue = True

    while Continue:

        if epoch >= 1 and epoch <= args.WARMUP_EPOCH:

            print('=' * 80)
            print('starting from WARMUP Phase')
            print('=' * 80)

            for name, param in model_train.named_parameters():

                if 'noise' in name:

                    param.requires_grad = False

                else:

                    param.requires_grad = True

            general_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_train.parameters()),
                                                 lr=warmup_lr,
                                                 betas=(0.9, 0.999))

            for warmup_epoch in range(epoch, args.WARMUP_EPOCH + 1):

                loss = train(model_train,
                             train_set,
                             warmup_epoch,
                             general_optimizer,
                             batch_num,
                             args.patch_size,
                             phase='WARMUP',
                             logger=logger,
                             entropy_term=args.entropy_term)

                psnr_ave, ssim_ave, psnr_std, ssim_std = test(test_set,
                                                             epoch,
                                                             logger,
                                                             args.patch_size,
                                                             mask_mode='randomcrop',
                                                             use_mask=mask3d_batch_randomcrop
                                                             )
                psnr_mean, ssim_mean = test(test_set,
                                             epoch,
                                             logger,
                                             args.patch_size,
                                             mask_mode='real256',
                                             use_mask=mask3d_batch_real256
                                             )

                psnr_max = save(psnr_ave,psnr_max,args.psnr_set,epoch, model_path,logger)

                print('>>>global_epoch = ', epoch)

                epoch += 1

            if epoch == args.WARMUP_EPOCH + 1:

                checkpoint(args.WARMUP_EPOCH, model_path, logger)

                print('=' * 80)
                print('WARMUP Phase finished, ckpt saved')
                print('=' * 80)

        current_epoch = (epoch - 1 - args.WARMUP_EPOCH) % (args.RECON_INTER + args.NOISE_INTER)
        print('current_epoch = ', current_epoch)


        # if current_epoch<= RECON_INTER and current_epoch>0:
        if current_epoch >= 0 and current_epoch < args.RECON_INTER:

            print('first do recon')

            for name, param in model_train.named_parameters():

                if 'noise' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

            recon_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_train.parameters()),
                                               lr=recon_lr,
                                               betas=(0.9, 0.999))

            # for i in range(current_epoch, RECON_INTER+1):
            for i in range(current_epoch + 1, args.RECON_INTER + 1):

                loss = train(model_train,
                             train_set,
                             i,
                             recon_optimizer,
                             batch_num,
                             args.patch_size,
                             phase='RECON',
                             logger=logger,
                             entropy_term=args.entropy_term)

                psnr_ave, ssim_ave, psnr_std, ssim_std = test(test_set,
                                                             epoch,
                                                             logger,
                                                             args.patch_size,
                                                             mask_mode='randomcrop',
                                                             use_mask=mask3d_batch_randomcrop)

                psnr_mean, ssim_mean = test(test_set,
                                             epoch,
                                             logger,
                                             args.patch_size,
                                             mask_mode='real256',
                                             use_mask=mask3d_batch_real256)

                psnr_max = save(psnr_ave, psnr_max, args.psnr_set, epoch, model_path, logger)

                print('>>>global_epoch = ', epoch)

                epoch += 1
                recon_epoch_sum += 1

                print('recon ++ || recon_epoch_sum=', recon_epoch_sum, '|| noise_epoch_sum=', noise_epoch_sum)


        else:

            print('then do noise')

            for name, param in model_train.named_parameters():

                if 'noise' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            noise_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_train.parameters()),
                                               lr=noise_lr,
                                               betas=(0.9, 0.999),
                                               weight_decay=1e-4)

            for j in range(current_epoch - args.RECON_INTER + 1, args.NOISE_INTER + 1):

                loss = train(model_train,
                             val_set,
                             j,
                             noise_optimizer,
                             batch_num,
                             args.noise_patch_size,
                             phase='NOISE',
                             logger=logger,
                             entropy_term=args.entropy_term)

                psnr_ave, ssim_ave, psnr_std, ssim_std = test(test_set,
                                                             epoch,
                                                             logger,
                                                             args.patch_size,
                                                             mask_mode='randomcrop',
                                                             use_mask=mask3d_batch_randomcrop)
                psnr_mean, ssim_mean = test(test_set,
                                         epoch,
                                         logger,
                                         args.patch_size,
                                         mask_mode='real256',
                                         use_mask=mask3d_batch_real256)

                psnr_max = save(psnr_ave, psnr_max, args.psnr_set, epoch, model_path, logger)


                print('>>>global_epoch = ', epoch)

                epoch += 1
                noise_epoch_sum += 1

                print('noise++ || recon_epoch_sum=', recon_epoch_sum, '|| noise_epoch_sum=', noise_epoch_sum)

        # recon_lr & noise_lr schedule
        if recon_epoch_sum % args.recon_lr_epoch == 0:
            recon_lr = recon_lr * args.recon_lr_scale
        if noise_epoch_sum % args.noise_lr_epoch ==0:
            noise_lr = noise_lr * args.noise_lr_scale

        # check the stopping criteria
        if recon_epoch_sum <= args.stop_criteria:

            Continue = True

        else:

            print('In summary || recon_epoch_sum=', recon_epoch_sum, '|| noise_epoch_sum=', noise_epoch_sum)

            Continue = False

    print('='*80)
    print('TRAINING END, START TESTING')
    print('='*80)

    psnr_ave, ssim_ave, psnr_std, ssim_std = test(test_set,
                                                 epoch-1,
                                                 logger,
                                                 args.patch_size,
                                                 mask_mode='randomcrop',
                                                 use_mask=mask3d_batch_randomcrop)

    psnr_mean, ssim_mean = test(test_set,
                             epoch - 1,
                             logger,
                             args.patch_size,
                             mask_mode='real256',
                             use_mask=mask3d_batch_real256)


if __name__ == '__main__':
    main(args.warmup_lr,
         args.recon_lr,
         args.noise_lr)
    

