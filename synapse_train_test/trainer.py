import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.colors import ListedColormap
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm


from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator

import matplotlib.pyplot as plt
import pandas as pd
import datetime
from datetime import datetime
import pytz

def inference(model, testloader, args, test_save_path=None):
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        logging.info(' idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(testloader.dataset)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return performance, mean_hd95


def plot_result(dice, h, snapshot_path,args):
    dict = {'mean_dice': dice, 'mean_hd95': h}
    df = pd.DataFrame(dict)
    plt.figure(0)
    df['mean_dice'].plot()
    resolution_value = 1200
    plt.title('Mean Dice')
    date_and_time = datetime.datetime.now()
    filename = f'{args.model_name}_' + str(date_and_time)+'dice'+'.png'
    save_mode_path = os.path.join(snapshot_path, filename)
    plt.savefig(save_mode_path, format="png", dpi=resolution_value)
    plt.figure(1)
    df['mean_hd95'].plot()
    plt.title('Mean hd95')
    filename = f'{args.model_name}_' + str(date_and_time)+'hd95'+'.png'
    save_mode_path = os.path.join(snapshot_path, filename)
    #save csv
    filename = f'{args.model_name}_' + str(date_and_time)+'results'+'.csv'
    save_mode_path = os.path.join(snapshot_path, filename)
    df.to_csv(save_mode_path, sep='\t')


def trainer_synapse(args, model, snapshot_path):
    global eta_min, T_mult
    os.makedirs(os.path.join(snapshot_path, 'test'), exist_ok=True)
    test_save_path = os.path.join(snapshot_path, 'test')
    logging.basicConfig(filename=f"{snapshot_path}/log_batch_{args.batch_size}.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    y_transforms = transforms.ToTensor()
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",img_size=args.img_size,
                               norm_x_transform = x_transforms, norm_y_transform = y_transforms)
    print("The length of train set is: {}".format(len(db_train)))
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    db_test = Synapse_dataset(base_dir=args.test_path, split="test_vol", list_dir=args.list_dir, img_size=args.img_size)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.AdamW(model.parameters(),lr = 0.001,betas = (0.9,0.999),eps = 1e-8 ,weight_decay = 1e-2 ,amsgrad = False)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    dice_=[]
    hd95_= []
    iterator = tqdm(range(max_epoch), ncols=70)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0= 50, T_mult=2, eta_min=1e-6,last_epoch = -1 )
    multi_list = []
    train_loss = []
    epoch_train_losses = []
    for epoch_num in iterator:
        total_loss = 0.0
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.squeeze(1).cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if (i_batch + 1) % 100 == 0:  
                current_lr = scheduler.get_last_lr()[0]
                print(f'Epoch {epoch_num + 1}, Batch {i_batch + 1}, Learning Rate: {current_lr}')
            total_loss += loss.item()
            iter_num = iter_num + 1
            # writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))



        # Calculate the Average Loss per Epoch
        rounded_total_loss = round(total_loss / (i_batch + 1), 4)
        epoch_train_losses.append(rounded_total_loss)
        writer.add_scalar('info/average_loss', rounded_total_loss, epoch_num)
        logging.info('Epoch %d - Average Loss: %f', epoch_num + 1, rounded_total_loss)
        # Test
        eval_interval = args.eval_interval
        current_time = datetime.now()
        formatted_time = current_time.strftime('%Y%m%d_%H%M')
        if epoch_num >= 249 and (epoch_num + 1) % 5 == 0 and epoch_num < 280:
            filename = f'{args.model_name}_epoch_{epoch_num}_bs_{args.batch_size}_{formatted_time}.pth'
            save_mode_path = os.path.join(snapshot_path, filename)
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            logging.info("*" * 20)

    # Plot and save the training loss curve.
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_train_losses, label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(snapshot_path, 'training_loss.png'))
    plt.show()
    plt.close()
    writer.close()
    return "Training Finished!"
