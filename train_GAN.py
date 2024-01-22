import argparse
import datetime
import json
import os
import time
import gc
import string
import random
import utils.auxiliaries as aux
import imgaug as ia
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from models.UnetGAN import UNet, init_weights
from models.UnetGAN import Critic, gradient_penalty, PerceptualLoss
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
# multiGPU
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

matplotlib.use('Agg')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
matplotlib.use('Agg')

'''an subprecess error happened when using DDP to train the WGAN with gradient pennalty, so use nn.dataparrallel instead'''


def main():
    '''---------------------------    Setup parser    ---------------------------'''
    parser = argparse.ArgumentParser()

    #-----------------------------   General Setup   ------------------------------#
    parser.add_argument('--savepath', default=os.path.join(os.getcwd(), "results_new"),
                        help='Where to store the results.')
    parser.add_argument('--datafolder', default=os.path.join(os.getcwd(), 'data'),
                        help='Where the rawdata is located.')
    parser.add_argument('--save_ram', action='store_true',
                        help='Use pre-saved tmp folder instead. Use "save_tmp.py".')
    parser.add_argument('--cuda', dest='cuda', action='store_true',
                        help='Use cuda (default = off)')
    parser.add_argument('--devices', nargs='+', type=int, default=0,
                        help='Cuda device to use (default = 0)')

    #--------------------------   Training Parameters   ---------------------------#
    parser.add_argument('--nepochs', type=int, default=75,
                        help='Number of epochs to train for.')
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping strategy')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate, default=0.00001')
    parser.add_argument('--b1', type=float, default=0.5,
                        help='beta1 parameter of Adam.')
    parser.add_argument('--b2', type=float, default=0.999,
                        help='beta2 parameter of Adam.')
    parser.add_argument('--criterion', default='L1',
                        help='L1 or L2 loss')
    parser.add_argument('--mbs', type=int, default=32,
                        help='Mini-batch size')
    parser.add_argument('--n_workers', type=int, default=8,
                        help='Number of workers to use for preprocessing.')

    #----------------------------   Data Parameters   -----------------------------#
    parser.add_argument('--ntrain', type=int, default=36800, #36800
                        help='Number of training samples')
    parser.add_argument('--nval', type=int, default=4800,
                        help='Number of validation samples')
    parser.add_argument('--sample_strategy', default='dataset',
                        help='Sample strategy for the data. If "dataset", sample \
                              uniform w.r.t to datasets. If "uniform" sample \
                              uniform over all datasets (not recommended).')
    parser.add_argument('--augment', action='store_true',
                        help='Augment the data.')
    parser.add_argument('--pBlur', type=float, default=0.8,
                        help='Probability for blur.')
    parser.add_argument('--pAffine', type=float, default=0.8,
                        help='Probability for affine transformations.')
    parser.add_argument('--pMultiply', type=float, default=0.8,
                        help='Probability for multiplicative augmentation.')
    parser.add_argument('--pContrast', type=float, default=0.8,
                        help='Probability for contrast augmentations.')

    parser.add_argument('--init_patch', type=int, default=512,
                        help='Size of patch before data augmentation.')
    parser.add_argument('--final_patch', type=int, default=384,
                        help='Size of patch after data augmentation.')
    parser.add_argument('--normalization', default='global',
                        help='Data normalization. Global or local (not recommended).')

    #---------------------------   Network Parameters   ---------------------------#
    parser.add_argument('--init_type', default='kaiming',
                        help='Weight initialization.')
    parser.add_argument('--upmode', default='conv',
                        help='Upsample mode')
    parser.add_argument('--downmode', default='sample',
                        help='Downsample mode')
    parser.add_argument('--batchnorm', action='store_true',
                        help='Use Batch-Normalization')
    parser.add_argument('--norm', default='an',
                        help='Normalization. Must be an, in or bn')
    parser.add_argument('--chs', nargs='+', type=int, default=0,
                        help='Channels to use for U-net')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Use dropout with provided probability.')
    parser.add_argument('--use_trained_model', action='store_true',
                        help='Use a trained model for model initialization.')
    parser.add_argument('--pretrained_model', default='DeepDSA_2023_03_12__133159_jnml',
                        help='pretrained model name')

    #--------------------------   Adversarial Training   --------------------------#
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Weight of perc. loss.')
    parser.add_argument('--beta', type=float, default=10,
                        help='Weight of pixelwise loss.')
    parser.add_argument('--lam', type=float, default=10.,
                        help='Probability for contrast augmentations.')
    parser.add_argument('--decay_iters', type=int, default=30000,
                        help='Iters after which to perform learning rate decay.')
    parser.add_argument('--critic_norm', default='in',
                        help='Normalization for critic. Must be an, in or bn')
    #------------------------------   Random Seeds   ------------------------------#
    parser.add_argument('--seed', type=int, help='manual seed')
    opt = parser.parse_args()

    # Setup device(s) to use
    if isinstance(opt.devices, list) and len(opt.devices) > 1:
       os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in opt.devices])
       opt.devices = list(range(len(opt.devices)))
    elif isinstance(opt.devices, list) and len(opt.devices) == 1:
       os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.devices[0])
       opt.devices = 0
    else:
       os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.devices)
       opt.devices = 0
    device = torch.device("cuda" if opt.cuda else "cpu")

    #------------------------------   Setup seeds   -------------------------------#
    if opt.seed is None:
        np.random.seed()
        opt.seed = np.random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    ia.seed(opt.seed)
    torch.backends.cudnn.benchmark = True

    #----------------------   Write Hyperparameters file   ------------------------#
    rundate = datetime.datetime.now()
    savetime = "{:02d}_{:02d}_{:02d}__{:02d}{:02d}{:02d}".format(
        rundate.year, rundate.month, rundate.day, rundate.hour, rundate.minute, rundate.second)
    random_str = "".join([random.choice(string.ascii_lowercase) for _ in range(4)])
    #-----------------------------   Setup Logger   -------------------------------#
    opt.savepath = os.path.join(opt.savepath, "DeepDSA_" + savetime + "_" + random_str)
    if not os.path.exists(opt.savepath):
        os.makedirs(opt.savepath)
    logger = SummaryWriter(log_dir=opt.savepath)

    '''----------------------------   Load Dataset    ---------------------------'''
    print('Setup dataloader...')
    start = time.time()

    Data_Train = aux.Data(opt, mode='train', normalization={
                              'type': opt.normalization, 'mean_std': None})
    Dataloader_Train = DataLoader(Data_Train, batch_size=opt.mbs, shuffle=True,
                                    num_workers=opt.n_workers, worker_init_fn=aux.worker_init_fn, pin_memory=True)
    Train_mean_std = Data_Train.normalization['mean_std'] if opt.normalization == 'global' else None
    opt.mean_std = [
        float(x) for x in Train_mean_std] if opt.normalization == 'global' else None

    Data_Val = aux.Data(opt, mode='val', normalization=Data_Train.normalization)
    Dataloader_Val = DataLoader(Data_Val, batch_size=opt.mbs, shuffle=True,
                                    num_workers=opt.n_workers, worker_init_fn=aux.worker_init_fn, pin_memory=True)
    print('...finished')
    stop = time.time()
    time_elapsed = stop - start
    print('Dataloader complates in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    with open(os.path.join(opt.savepath, 'Hyperparameters.txt'), 'w') as f:
        json.dump(vars(opt), f)

    '''-----------------------------------------------------------------------------
    --------------------   Setup network, loss & optimizers   ----------------------
    -----------------------------------------------------------------------------'''
    print('Setup the network and loss...')
    #-----------------------------   Setup network  -------------------------------#
    if opt.chs == 0:
        ch = [32, 64, 128, 256, 512]
    else:
        ch = opt.chs
    print('Setup Unet with chs: ', ch)

    net = UNet(ch=ch, downmode=opt.downmode,
               upmode=opt.upmode, dropout=opt.dropout)
    critic = Critic(opt.final_patch, in_ch=2, input_ft=32, depth=5, max_ft=512,
                    norm=opt.critic_norm)
    if opt.use_trained_model:
        print("use trained model from ", opt.pretrained_model, " as initialization model")
        checkpoint = torch.load(
            os.path.join('\\results', opt.pretrained_model, "best_train_net_G.pt"))
        net.load_state_dict(checkpoint['model_G'])
        critic.load_state_dict(checkpoint['model_D'])

    else:
        init_weights(net, init_type=opt.init_type)

    net.to(device)
    critic.to(device)
    if isinstance(opt.devices, list):
        net = nn.DataParallel(net, device_ids=opt.devices)
        critic = nn.DataParallel(critic, device_ids=opt.devices)
    #------------------------------   Setup loss  ---------------------------------#
    if opt.criterion == 'L1':
        pix_crit = nn.L1Loss()
    elif opt.criterion == 'L2':
        pix_crit = nn.MSELoss()

    #---------------------------   Setup optimizers  ------------------------------#
    n_dsteps = 4
    g_optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    d_optimizer = optim.Adam(critic.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    print('...finished')

    '''-----------------------------------------------------------------------------
    -----------------------------   Train network    -------------------------------
    -----------------------------------------------------------------------------'''
    print('Start training...')
    start = time.time()
    total_iters = 0
    best_val_loss_G = np.inf
    best_train_loss_G = np.inf
    best_val_loss_D = np.inf
    best_train_loss_D = np.inf
    for epoch in range(opt.nepochs):
        torch.manual_seed(opt.seed + epoch)
        np.random.seed(opt.seed + epoch)

        train_losses_log = {l: 0.0 for l in ["d_loss", "grad_p_loss", "g_loss_perc", "g_loss_pix", "g_loss_adv"]}
        val_losses_log = {l: 0.0 for l in ["d_loss", "grad_p_loss", "g_loss_perc", "g_loss_pix", "g_loss_adv"]}

        '''-------------------------    training    -------------------------'''
        net.train()
        critic.train()
        for i_batch, sample_batched in enumerate(Dataloader_Train):
            total_iters += 1
            input, target = Variable(sample_batched['x']).to(device, non_blocking=True), \
                            Variable(sample_batched['y']).to(device, non_blocking=True)

            #  Train Critic
            d_optimizer.zero_grad()
            for _ in range(n_dsteps):
                fake = net(input)
                fake_c = torch.cat([input, fake], dim=1)
                target_c = torch.cat([input, target], dim=1)
                grad_p = gradient_penalty(device, critic, target_c, fake_c, lam=opt.lam)
                critic_loss = torch.mean(critic(fake_c)) - torch.mean(critic(target_c))
                loss_D = critic_loss + grad_p

                loss_D.backward()
                d_optimizer.step()

            #  Train Generator
            g_optimizer.zero_grad()

            fake = net(input)
            fake_c = torch.cat([input, fake], dim=1)
            loss_G_adv = -torch.mean(critic(fake_c))
            loss_G_perc = 0.0  #perceptual(fake, target)
            Loss_G_pix = pix_crit(fake, target)
            loss_G = opt.beta * Loss_G_pix + loss_G_adv
            loss_G.backward()
            g_optimizer.step()

            train_losses_log['d_loss'] += loss_D.data.item()
            train_losses_log['grad_p_loss'] += grad_p.item()
            train_losses_log['g_loss_perc'] += 0.
            train_losses_log['g_loss_pix'] += Loss_G_pix.item()
            train_losses_log['g_loss_adv'] += loss_G_adv.item()

            print('[TRAIN: epoch %2d of %2d | minibatch %3d of %3d | loss G %.4f | loss D %.4f]' % (epoch + 1, opt.nepochs, i_batch + 1,
                  len(Dataloader_Train), loss_G.data.item(), loss_D.data.item()))

            if i_batch == 0:
                ''' an issue happened to add_graph when nn.dataparallel'''
                print('save patches...')
                patches = {'input': input[:6, 0, :, :].data.cpu().numpy(),
                           'target': target[:6, 0, :, :].data.cpu().numpy(),
                           'output': fake[:6, 0, :, :].data.cpu().numpy()}

            del input, target, loss_G, loss_D
            gc.collect()
            torch.cuda.empty_cache()

        for l in train_losses_log:
            train_losses_log[l] /= len(Dataloader_Train)

        '''-------------------------    validation    -------------------------'''
        net.eval()
        critic.eval()
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(Dataloader_Val):
                input, target = Variable(sample_batched['x']).to(device, non_blocking=True), \
                                Variable(sample_batched['y']).to(device, non_blocking=True)

                fake = net(input)
                fake_c = torch.cat([input, fake], dim=1)
                target_c = torch.cat([input, target], dim=1)
                grad_p = grad_p  # element 0 of tensors does not require grad and does not have a grad_fn
                critic_loss = torch.mean(critic(fake_c)) - torch.mean(critic(target_c))
                loss_D = critic_loss + grad_p

                loss_G_adv = -torch.mean(critic(fake_c))
                loss_G_perc = 0.0  # perceptual(fake, target)
                Loss_G_pix = pix_crit(fake, target)
                loss_G = opt.beta * Loss_G_pix + loss_G_adv

                val_losses_log['d_loss'] += loss_D.data.item()
                val_losses_log['grad_p_loss'] += grad_p.data.item()
                val_losses_log['g_loss_perc'] += 0.  # loss_G_perc.item()
                val_losses_log['g_loss_pix'] += Loss_G_pix.item()
                val_losses_log['g_loss_adv'] += loss_G_adv.item()

                del input, target, loss_G, loss_D
                gc.collect()
                torch.cuda.empty_cache()

        for l in val_losses_log:
            val_losses_log[l] /= len(Dataloader_Val)
        print('[VAL: epoch %2d of %2d | loss G %.4f | loss D %.4f]' % (
            epoch + 1, opt.nepochs, val_losses_log['g_loss_adv'], val_losses_log['d_loss']))

        '''--------------------------    Save Logs    ---------------------------'''
        stop = time.time()
        aux.write_log(os.path.join(opt.savepath, "Log_g.csv"), epoch, stop -
                      start, g_optimizer.param_groups[0]['lr'], train_losses_log['g_loss_pix'], val_losses_log['g_loss_pix'])
        aux.make_learning_curves_fig(os.path.join(opt.savepath, "Log_g.csv"), att='_g')

        aux.write_log(os.path.join(opt.savepath, "Log_d.csv"), epoch, stop -
                      start, g_optimizer.param_groups[0]['lr'], train_losses_log['d_loss'], val_losses_log['d_loss'])
        aux.make_learning_curves_fig(os.path.join(opt.savepath, "Log_d.csv"), att='_d')

        '''------------------------    Save Networks    ------------------------'''
        if train_losses_log['d_loss'] < best_train_loss_D:
            best_train_loss_D = train_losses_log['d_loss']
            state_train = {'model_G': net.state_dict(), 'model_D': critic.state_dict(),
                               'normalization': Data_Train.normalization}
            torch.save(state_train, os.path.join(
                opt.savepath, 'best_train_net_D.pt'))

        if val_losses_log['d_loss'] < best_val_loss_D:
            best_val_loss_D = val_losses_log['d_loss']
            state_train = {'model_G': net.state_dict(), 'model_D': critic.state_dict(),
                               'normalization': Data_Train.normalization}
            torch.save(state_train, os.path.join(
                opt.savepath, 'best_val_net_D.pt'))

        if train_losses_log['g_loss_pix'] < best_train_loss_G:
            best_train_loss_G = train_losses_log['g_loss_pix']
            print('Apply to train stacks')
            train_log_imgs = aux.apply_to_stacks(
                Data_Train, [0, 5, 6], net, epoch, opt, Data_Train.normalization)#0,5,6
            state_train = {'model_G': net.state_dict(), 'model_D': critic.state_dict(),
                               'normalization': Data_Train.normalization}
            torch.save(state_train, os.path.join(
                opt.savepath, 'best_train_net_G.pt'))

            logger.add_images('train applied', aux.min_max_norm(np.expand_dims(np.stack(
                train_log_imgs['applied'], 0), 1)), epoch + 1)
            if epoch == 0:
                logger.add_images('train target', aux.min_max_norm(np.expand_dims(np.stack(
                    train_log_imgs['target'], 0), 1)), epoch + 1)
        if val_losses_log['g_loss_pix'] < best_val_loss_G:
            best_val_loss_G = val_losses_log['g_loss_pix']
            print('Apply to validation stacks')
            val_log_imgs = aux.apply_to_stacks(
                Data_Val, [0, 1, 2], net, epoch, opt, Data_Train.normalization)
            state_train_val = {'model_G': net.state_dict(), 'model_D': critic.state_dict(),
                               'normalization': Data_Train.normalization}
            torch.save(state_train_val, os.path.join(
                opt.savepath, 'best_val_net_G.pt'))

            logger.add_images('val applied', aux.min_max_norm(np.expand_dims(np.stack(
                val_log_imgs['applied'], 0), 1)), epoch + 1)
            if epoch == 0:
                logger.add_images('val target', aux.min_max_norm(np.expand_dims(np.stack(
                    val_log_imgs['target'], 0), 1)), epoch + 1)
        '''----------------------    Tensorboard Logs    --------------------------'''
        # Tensorboard Logging
        losses = {'train ': train_losses_log, 'val ': val_losses_log}
        for tag_, value in losses.items():
            for tag, value in value.items():
                logger.add_scalar(tag_+tag, value, epoch + 1)

        logger.add_images('patch input',  aux.min_max_norm(np.expand_dims(patches['input'], 1)), epoch + 1)
        logger.add_images('patch target', aux.min_max_norm(np.expand_dims(patches['target'], 1)), epoch + 1)
        logger.add_images('patch output', aux.min_max_norm(np.expand_dims(patches['output'], 1)), epoch + 1)

if __name__ == '__main__':
    main()