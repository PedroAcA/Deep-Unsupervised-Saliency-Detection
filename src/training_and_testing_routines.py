import os
import warnings
import matplotlib.pyplot as plt
import torch
import logging
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
from utils.early_stopping import EarlyStopping
from utils.metrics import log_metrics
from utils.meter import MetricLogger

def train(cfg, model, optimizer, scheduler, poly_decay, loader, chkpt, writer, offset):
    """
    Method for training deep unsupervised sailency detection model.
    Training module implements 4 training methodology
    1. Real: The backbone network is trainined with Ground Truth object, loss = mean((pred_y_i, y_gt_i))
    3. Noise: The backbone network is trained using losses on all the labels of unsupersived methods, loss = mean((pred_y_i, unup_y_i_m))
    3. Avg: The backbone network is trained  using avg of the all the unsupervised methods as ground truth, loss = mean((pred_y_i, mean(unup_y_i_m)))
    4. Full: The backbone network as well as the noise module is trained, the training proceeds as follows
    Training processeds in rounds,
       Round 1:
         Initalise the variane of the prior noise = 0.0
         Train the backbone network on all the unsuperversived methods loss = bcewithlogitloss(pred_y, unsup_y_i)
         Once the network is trained till converge
         Update the noise network using the update rule in Eq 7 in the paper
    Round i:
        Sample noise from the noise network
        Training the backbone network on all the unsupervised methods loss = bcewithlogitloss(pred_y + noise, unsup_y_i) + noise loss computed using Eq 6
        Train the backbone network till convergence
        Update the noise network using the update rule in Eq 7
    Args
    ---
    cfg: (CfgNode) Master configuration file
    model: (tuple) Consits of 2 model the backbone network and the noise module.
    optimizer: torch.optim Optimizer to train the network
    scheduler: torch.optim.lr_scheduler Scheduler for fixing learning rate
    loader: (tuple) traindataset loader and validation dataset loader
    chkpt: chekpointer
    writer: tensorboard writer
    offset: start point to save the model correctly
    """
    device = cfg.SYSTEM.DEVICE
    logger = logging.getLogger(cfg.SYSTEM.EXP_NAME)
    h, w = cfg.SOLVER.IMG_SIZE[0], cfg.SOLVER.IMG_SIZE[1]
    batch_size = cfg.SOLVER.BATCH_SIZE
    train_loader, val_loader = loader
    num_batches = len(train_loader.dataset) // batch_size
    pred_criterion = torch.nn.BCELoss()
    pred_model, noise_model = model
    early_stopping = EarlyStopping(pred_model, noise_model, val_loader, cfg)
    writer_idx = 0
    use_validation = cfg.DATA.VAL.NOISE_ROOT or cfg.DATA.VAL.GT_ROOT
    for epoch in range(cfg.SOLVER.EPOCHS):
        pred_model.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_loader):
            writer_idx = batch_idx * batch_size + (epoch * num_batches * batch_size)
            item_idxs = data['idx']
            # x: Input data, (None, 3, H, W)
            x = data['sal_image'].to(device)
            # y: GT label (None, 1, H, W), required for metrics
            y = data['sal_label'].to(device)
            orig_x = x  # used for computing metrics upon true data
            if cfg.SYSTEM.EXP_TYPE == 'noise':
                y_noise = data['sal_noisy_label'].to(device)
                # x: repeat input for each map, (None*NUM_MAPS, 3, H, W) !Note: Take care of batch size
                x = torch.repeat_interleave(x, repeats=cfg.SOLVER.NUM_MAPS, dim=0)
                # y_noise: Unsupervised labels, (None*NUM_MAPS, 1, H, W)
                y_noise = torch.reshape(y_noise, (batch_size*cfg.SOLVER.NUM_MAPS, 1, h, w))
            elif cfg.SYSTEM.EXP_TYPE == 'avg':
                y_noise = data['sal_noisy_label'].to(device)
                # y_noise: Taking mean of all the noise labels, (None, 1, H, W)
                y_noise = torch.mean(y_noise, dim=1, keepdim=True)
            elif cfg.SYSTEM.EXP_TYPE == 'real':
                # y: GT label, (None, 1, H, W)
                y_noise = y
            else:
                y_noise = data['sal_noisy_label'].to(device)
                y_noise = torch.reshape(y_noise, (batch_size, cfg.SOLVER.NUM_MAPS, h, w))

            pred = pred_model(x)
            if cfg.SYSTEM.EXP_TYPE == 'full':
                y_pred = noise_model.add_noise_to_prediction(pred, item_idxs)
            else:
                y_pred = pred

            # Computing BCE Loss between pred and target, See Eq 4
            pred_loss = pred_criterion(y_pred, y_noise)
            noise_loss = 0.0
            if cfg.SYSTEM.EXP_TYPE == 'full':
                noise = y_noise - pred
                # compute noise loss for batch using Eq 6
                emp_std = torch.std(noise, 1, unbiased=True)
                emp_mean = torch.zeros_like(emp_std) # we assume errors are zero mean gaussian
                emp_std = torch.clamp(emp_std, min=noise_model.small_ct)
                emp_dist = Normal(emp_mean, emp_std)
                prior_dist = noise_model.get_batch_prior_distribution(item_idxs)
                noise_loss = torch.sum(kl_divergence(prior_dist, emp_dist))

            # total loss computed using Eq 2, this loss is used only for training the backbone network parameters
            total_loss = pred_loss + cfg.SOLVER.LAMBDA * noise_loss
            total_loss.backward()
            if ((batch_idx+1)%cfg.SOLVER.ITER_SIZE == 0) or ((batch_idx+1) == len(train_loader)): #code for batch accumulation from https://stackoverflow.com/questions/68479235/cuda-out-of-memory-error-cannot-reduce-batch-size (accessed on 24th of Sept, 2021)
                optimizer.step()
                optimizer.zero_grad()

            if batch_idx % cfg.SYSTEM.LOG_FREQ == 0 and cfg.SYSTEM.EXP_TYPE == 'real':
                logger.info(f'epoch:{epoch}, batch_idx:{batch_idx},Noise Loss:{noise_loss},Pred Loss: {pred_loss}')
                writer.add_scalar('Loss', pred_loss, writer_idx)
                # compute metrics
                y_pred_t = pred_model(orig_x)
                metrics = log_metrics(y_pred_t, y)
                writer.add_scalars("metrics", {"precision": metrics.meters['precision'].avg,
                                               "recall": metrics.meters['recall'].avg,
                                               "mae": metrics.meters['mae'].avg}, writer_idx)
            elif batch_idx % cfg.SYSTEM.LOG_FREQ == 0:
                logger.info(f'epoch:{epoch}, batch_idx:{batch_idx},\
                            Lambda*Noise Loss:{cfg.SOLVER.LAMBDA*noise_loss}, Pred Loss: {pred_loss}')
                print(f'epoch:{epoch}, batch_idx:{batch_idx},\
                        Lambda*Noise Loss:{cfg.SOLVER.LAMBDA*noise_loss}, Pred Loss: {pred_loss}')

        if use_validation:
            early_stopping.validate(writer, writer_idx)

        poly_decay.step()
        if use_validation:
            scheduler.step(early_stopping.val_loss.meters['val_loss'].avg)

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], writer_idx)
        print(f"lr: {optimizer.param_groups[0]['lr']}")

        if early_stopping.converged or (epoch+1) ==  cfg.SOLVER.EPOCHS or ((epoch+1) % cfg.SYSTEM.CHKPT_FREQ == 0 ):
            fn = f'checkpoint_epoch_{cfg.SYSTEM.EXP_NAME}_{epoch + offset}'
            chkpt.save(fn, epoch=epoch)

        if cfg.SYSTEM.EXP_TYPE in ['real', 'avg', 'noise']:
            if early_stopping.converged:
                logger.info("Converged")
                return
        else:
            if early_stopping.converged:
                # Reset Early Stopping variables, training till next convergence
                early_stopping.reset()
                #Decrease base learning rate to propagate the Reduce on LR Plateau effects to the poly decay policy
                poly_decay.base_lrs[0] = cfg.SOLVER.FACTOR*poly_decay.base_lrs[0]
                # Update the noise variance using Eq 7. !Note: Importantly we do this for all images encountered, using the pred_variance
                logger.info('Updating Noise Variance')
                noise_model.update(pred_model, train_loader)


def create_folder(folder):
    os.makedirs(folder, exist_ok=True)


def save_img_tensor(config, img_tensor, filename, title='', grayscale=False, remove_xy_ticks=True, cmap='seismic',
                    alpha=None, dpi=300):
    folder = config.SAVE_ROOT + 'results/' + config.SYSTEM.EXP_NAME + '/' + 'qualitative/'
    num_of_dims = len(img_tensor.shape)
    assert 2 <= num_of_dims <= 3, "Expected tensor of dimension 2 or 3, but got {}".format(num_of_dims)
    create_folder(folder)
    print("Saving {} to {}".format(filename, folder))
    if grayscale:
        cmap = 'gray'

    if num_of_dims == 2:
        plt.imshow(img_tensor.detach().cpu(), cmap=cmap, alpha=alpha)
    else:
        plt.imshow(img_tensor.detach().cpu().permute(1, 2, 0), cmap=cmap,
                   alpha=alpha)  # permute from C,H,W to H,W,C and show image

    plt.title(title)
    if remove_xy_ticks:
        plt.xticks([])
        plt.yticks([])

    plt.savefig(folder + filename, bbox_inches='tight', dpi=dpi)
    print("Saved {} to {}".format(filename, folder))
    plt.figure()


def test(cfg, model, loader, test_type, logger, samples=None):
    device = cfg.SYSTEM.DEVICE
    if test_type != 'qualitative':
        metrics = MetricLogger()

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
                if test_type != 'quantitative' and samples and data['im_name'][0][:-4] in samples:
                    save_imgs = True
                elif test_type != 'quantitative' and not samples:
                    save_imgs = True
                    warnings.warn('No samples provided. Saving qualitative results for the whole testing set')
                else:
                    save_imgs = False

                print("Batch idx {}".format(batch_idx))
                model.eval()
                # writer_idx = batch_idx * batch_size + num_batches * batch_size
                # import ipdb; ipdb.set_trace()
                x = data['image'].to(device)
                y = data['label'].to(device)
                pred = model(x)
                bin_th = 0.5
                if test_type != 'qualitative':
                    log_metrics(pred, y, metrics, bin_th=bin_th)

                if test_type != 'quantitative' and save_imgs:
                    image_filename = data['im_name'][0][:-4] + '.png'
                    save_img_tensor(cfg, pred[0, 0], 'pred_' + image_filename, grayscale=True, remove_xy_ticks=True)
                    bin_pred = (pred > bin_th).to(torch.float)
                    save_img_tensor(cfg, bin_pred[0, 0], 'bin_pred_' + image_filename, grayscale=True, remove_xy_ticks=True)
                    save_img_tensor(cfg, y[0, 0], 'gt_' + image_filename, grayscale=True, remove_xy_ticks=True)

    if test_type != 'qualitative':
        logger.info(str(metrics))
