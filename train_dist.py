import argparse

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torch.distributed as dist
import torchvision.transforms as transforms
import os, sys
import os.path as osp
# import logging
import json
from tqdm import tqdm
import numpy as np
import collections
from src.datasets.subCOCO import CocoSubDetection

from torch.optim import lr_scheduler
from src.helper_functions.helper_functions import mAP, CocoDetection, CutoutPIL, ModelEma, add_weight_decay, sl_mAP, pred_merger
from src.helper_functions.logger import setup_logger
from src.models import create_model
from src.loss_functions.losses import KMCL_Loss
from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast
from Kmcl_Class import KMCL
from src.datasets.VOC import VOC2007
from src.datasets.ADPDataset import ADPDataset
from meters import *
import pandas as pd
# torch.multiprocessing.set_sharing_strategy('file_system')

# Kill Command: kill $(ps aux | grep train_dist.py | grep -v grep | awk '{print $2}')
# Launch Command: CUDA_VISIBLE_DEVICES=0,1,2,3 NCLL_DEBUG=INFO python -m torch.distributed.launch --nproc_per_node=4 train_dist.py 
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('--data_folder', help='path to dataset', default='/fs2/comm/kpgrp/mhosseini/project_MCL/')
parser.add_argument('--lr', default=4e-4, type=float)
parser.add_argument('--model-name', default='tresnet_l_v2')
parser.add_argument('--model-path', 
                    default='/fs2/comm/kpgrp/mhosseini/github/KMCL/networks/models/tresnet_l_v2_miil_21k.pth',
                    type=str)
parser.add_argument('--epochs', default=80)
parser.add_argument('--num-classes', default=80)
parser.add_argument('--dataset', 
                    choices=["PascalVOC", "COCO", "ADP", "Xray", "COCOSub"], default="COCO")
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--image-size', default=448, type=int,
                    metavar='N', help='input image size (default: 448)')

parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--print-freq', '-p', default=64, type=int,
                    metavar='N', help='print frequency (default: 64)')
parser.add_argument('--distributed', action='store_true', help='using dataparallel')
parser.add_argument('--dtgfl', action='store_true', 
            help='using disable_torch_grad_focal_loss in ASL loss')
parser.add_argument('--output',
                    help='path to output folder', default="newLogs/")

# distribution training
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')

parser.add_argument('--loss-opt', default="all", choices=["all", "ASLOnly"], help='loss type, only ASL vs ASL + KMCL + REC')

parser.add_argument('--loss-case', default="anisotropic", choices=["isotropic", "anisotropic"], help='loss case')

parser.add_argument('--similarity', default="BC", choices=["BC", "MKS", "RBF"], help='similarity metric')

parser.add_argument('--num-samples-sub', default=0, type=int, help='Only set for sub-sampling the Datasets, else 0')

def main():
    args = parser.parse_args()
    args.do_bottleneck_head = False
    if args.dataset == "ADP":
        num_classes = 9
        
    elif args.dataset == 'PascalVOC':
        num_classes = 20
        
    elif args.dataset == 'COCO':
        num_classes = 80
    
    elif args.dataset == 'Xray':
        num_classes = 14
    elif args.dataset == 'COCOSub':
        num_classes = 80
        assert args.num_samples_sub != 0
    else:
        raise NotImplementedError(f"{args.dataset} is not implemented")
    
    args.num_classes = num_classes
    
    # setup dist training
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank

    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        print('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.' % (args.rank, args.world_size))
    else:
        print('Training with a single process on 1 GPUs.')
    assert args.rank >= 0

    # setup logger
    logger = setup_logger(output=args.output, distributed_rank=dist.get_rank(), color=False, name="Coco")
    logger.info("Command: "+' '.join(sys.argv))
    if dist.get_rank() == 0:
        path = os.path.join(args.output, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(path))
        os.makedirs(osp.join(args.output, 'tmpdata'), exist_ok=True)

    # Setup model
    logger.info('creating model...')
    
    encoder_model, dim = create_model(args)
    encoder_model = encoder_model.cuda()
    
    if args.model_path:  # make sure to load pretrained ImageNet model
        state = torch.load(args.model_path, map_location='cpu')
        
        filtered_dict = {k: v for k, v in state['state_dict'].items() if
                         (k in encoder_model.state_dict() and 'head.fc' not in k)}
        encoder_model.load_state_dict(filtered_dict, strict=False)
    
    model = KMCL(encoder_model, dim, out_classes=args.num_classes,args=args)
    
    logger.info('done\n')

    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False\
                                                          ,find_unused_parameters=True)

    # COCO Data loading
    if args.dataset == "COCO":
        COCO_image_normalization_mean=[0.485, 0.456, 0.406]
        COCO_image_normalization_std=[0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=COCO_image_normalization_mean,
                                                std=COCO_image_normalization_std)
        instances_path_val = os.path.join(args.data_folder, 'coco/data/annotations/instances_val2014.json')
        instances_path_train = os.path.join(args.data_folder, 'coco/data/annotations/instances_train2014.json')
        data_path_val   = f'{args.data_folder}coco/data/'    # args.data
        data_path_train = f'{args.data_folder}coco/data'  # args.data
        val_dataset = CocoDetection(data_path_val,
                                    instances_path_val,
                                    transforms.Compose([
                                        transforms.Resize((args.image_size, args.image_size)),
                                        transforms.ToTensor(),
                                        normalize 
                                    ]))
        train_dataset = CocoDetection(data_path_train,
                                    instances_path_train,
                                    transforms.Compose([
                                        transforms.Resize((args.image_size, args.image_size)),
                                        CutoutPIL(cutout_factor=0.5),
                                        RandAugment(),
                                        transforms.ToTensor(),
                                        normalize
                                    ]))
        
    elif args.dataset == "PascalVOC":
        
        VOC_image_normalization_mean=[0.485, 0.456, 0.406]
        VOC_image_normalization_std=[0.229, 0.224, 0.225]
        
        normalize = transforms.Normalize(mean=VOC_image_normalization_mean,
                                                std=VOC_image_normalization_std)
        train_transform = transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    CutoutPIL(cutout_factor=0.5),
                                    RandAugment(),
                                    transforms.ToTensor(),
                                    normalize])
        
        test_transform = transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    normalize
                                    ])
        
        
        train_dataset = VOC2007(root=args.data_folder + "voc/", transform=train_transform, split='train')
        

        val_dataset = VOC2007(root=args.data_folder + "voc/", transform=test_transform, split='test')
        
        
    elif args.dataset == 'ADP':     
        ADP_image_normalization_mean=[0.81233799, 0.64032477, 0.81902153]
        ADP_image_normalization_std=[0.18129702, 0.25731668, 0.16800649]
            
        normalize = transforms.Normalize(mean=ADP_image_normalization_mean,
                                             std=ADP_image_normalization_std)
       
        train_transform = transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    CutoutPIL(cutout_factor=0.5),
                                    RandAugment(),
                                    transforms.ToTensor(),
                                    normalize])
        
        test_transform = transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    normalize
                                    ])
        
        train_dataset = ADPDataset(level='L1', root='/fs2/comm/kpgrp/mhosseini/project_MCL/', 
                            transform=train_transform, split='train')

        val_dataset = ADPDataset(level='L1', root='/fs2/comm/kpgrp/mhosseini/project_MCL/',
                            transform= test_transform, split='test')
        
    elif args.dataset == 'Xray': 
        mean = [0.50576189,0.50576189,0.50576189]
        normalize = transforms.Normalize(mean, [1.,1.,1.])
        
        train_transform = transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    CutoutPIL(cutout_factor=0.5),
                                    transforms.RandomAffine(45, translate=(0.15, 0.15), scale=(0.85, 1.15)),
                                    transforms.ToTensor(),
                                    normalize])
        test_transform = transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    normalize
                                  ])
        train_dataset = CXRDataset('/fs2/comm/kpgrp/mhosseini/project_MCL/Xray8/', transform=train_transform)
        val_dataset = CXRDataset('/fs2/comm/kpgrp/mhosseini/project_MCL/Xray8/', dataset_type='test', transform=test_transform)
    
    elif args.dataset == "COCOSub":
        COCO_image_normalization_mean=[0.485, 0.456, 0.406]
        COCO_image_normalization_std=[0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=COCO_image_normalization_mean,
                                                std=COCO_image_normalization_std)
        instances_path_val = os.path.join(args.data_folder, 'coco/data/annotations/instances_val2014.json')
        instances_path_train = os.path.join(args.data_folder, 'coco/data/annotations/instances_train2014.json')
        data_path_val   = f'{args.data_folder}coco/data/'    # args.data
        data_path_train = f'{args.data_folder}coco/data'  # args.data
        val_dataset = CocoSubDetection(data_path_val,
                                    instances_path_val,
                                    transforms.Compose([
                                        transforms.Resize((args.image_size, args.image_size)),
                                        transforms.ToTensor(),
                                        normalize 
                                    ]))
        train_dataset = CocoSubDetection(data_path_train,
                                    instances_path_train,
                                    transforms.Compose([
                                        transforms.Resize((args.image_size, args.image_size)),
                                        CutoutPIL(cutout_factor=0.5),
                                        RandAugment(),
                                        transforms.ToTensor(),
                                        normalize
                                    ]), num_samples = args.num_samples_sub)
        
        
    print("len(val_dataset)): ", len(val_dataset))
    print("len(train_dataset)): ", len(train_dataset))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    assert args.batch_size // dist.get_world_size() == args.batch_size / dist.get_world_size(), 'Batch size is not divisible by num of gpus.'

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size // dist.get_world_size(), 
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size // dist.get_world_size(), 
        shuffle=False,
        num_workers=args.workers, pin_memory=False, sampler=val_sampler)

    # Actuall Training
    train_multi_label(model, ema, train_loader, val_loader, args.lr, args, logger)


def train_multi_label(model, ema, train_loader, val_loader, lr, args, logger):
    # set optimizer
    if dist.get_rank() == 0:
        id = str(np.random.randint(100000))
        losses = collections.defaultdict(list)
    else:
        id = None
        losses = None
    Epochs = args.epochs
    weight_decay = 1e-4
    criterion = KMCL_Loss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True, loss_case=args.loss_case, loss_opt = args.loss_opt)
    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs, pct_start=0.2)

    highest_mAP = 0
    trainInfoList = []
    scaler = GradScaler()
    for epoch in range(Epochs):
        LossDict = {
            "ASLLoss":[],
            "NLLLoss" :[],
            "BDLoss" :[]
            }
        for i, (inputData, target) in enumerate(train_loader):
            # break
            inputData = inputData.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            Normlabels = target
            if args.dataset == "COCO":
                Normlabels = (target.max(dim=1)[0]).float()
                target = Normlabels
            elif args.dataset == "PascalVOC":
                Normlabels = (target >= 0).float().cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                
            with autocast():  # mixed precision
                
                features, gaussian_params = model(inputData)
                
            loss, ASLLoss, NLLLoss, BDLoss = criterion(features, gaussian_params, Normlabels)
            
            if dist.get_rank() == 0:
                LossDict["ASLLoss"].append(ASLLoss.detach().cpu())
                LossDict["NLLLoss"].append(NLLLoss.detach().cpu())
                LossDict["BDLoss"].append(BDLoss.detach().cpu())
            model.zero_grad()

            scaler.scale(loss).backward()
           

            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()

            ema.update(model)
            # store information
            if i % 100 == 0:
                trainInfoList.append([epoch, i, loss.item()])
                logger.info('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f} , ASLLoss: {:.1f} , KMCLLoss: {:.1f} , NLLLoss: {:.1f}'
                      .format(epoch, Epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3),
                              scheduler.get_last_lr()[0], \
                              loss.item(), ASLLoss.item(), BDLoss.item(), NLLLoss.item()))

        model.eval()
        if dist.get_rank() == 0:
            for i in ["ASLLoss", "BDLoss", "NLLLoss"]:
                localDict = [x for x in LossDict[i] if torch.isfinite(x).item() and x.item() != 0]
                if len(localDict) == 0:
                    mean = float('nan')
                elif len(localDict) == 1:
                    mean = localDict[0]
                else:
                    mean = torch.mean(torch.stack(localDict)).item()
                losses[i].append(mean)
                
        mAP_score_regular, mAP_score_ema = validate_multi(val_loader, model, ema, logger, args, losses, id)
        model.train()

            
        mAP_score = max(mAP_score_regular, mAP_score_ema)
        
        if dist.get_rank() == 0:
            if mAP_score > highest_mAP:
                highest_mAP = mAP_score
                print('ID = {}  | current_mAP = {:.2f}, highest_mAP = {:.2f}\n'.format(id, mAP_score, highest_mAP))
                try:
                    torch.save(model.state_dict(), os.path.join(
                        'saved_models/', 'model-ASL-{}_{}_ID_{}.ckpt'.format(args.dataset, args.model_name, id)))
                    
                    torch.save(ema.module.state_dict(), os.path.join(
                        'saved_models/', 'ema-model-ASL-{}_{}_ID_{}.ckpt'.format(args.dataset, args.model_name, id)))
                    
                    
                except:
                    pass

            print('ID = {}  | current_mAP = {:.2f}, highest_mAP = {:.2f}\n'.format(id, mAP_score, highest_mAP))
            highest_mAP = max(highest_mAP, mAP_score)
            logger.info('| current_mAP = {:.2f}, highest_mAP = {:.2f}\n'.format(mAP_score, highest_mAP))


def save_checkpoint(state_dict, savedir, savedname, is_best, rank=0):
    torch.save(state_dict, os.path.join(savedir, savedname))
    if is_best:
        torch.save(state_dict, os.path.join(savedir, 'model-highest.ckpt'))


def validate_multi(val_loader, model, ema_model, logger, args, losses, id):
    logger.info("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targets = []
    
    if dist.get_rank() == 0:
        batchs = tqdm(val_loader)
        de = False
        if args.dataset == "PascalVOC":
            de = True
        Modelmeter = initialize_meters(dist=False, difficult_example=de, ws=4)
        Modelmeter = on_start_epoch(Modelmeter)
        
        Emameter = initialize_meters(dist=False, difficult_example=de, ws=4)
        Emameter = on_start_epoch(Emameter)
    else:
        batchs = val_loader

    for i, (input, target) in enumerate(batchs):
        # target = target
        # target = target.max(dim=1)[0]
        # import ipdb; ipdb.set_trace()
        # compute output
        with torch.no_grad():
            with autocast():   
                output_pi = model(input.cuda())[1]["pi"]
                output_regular = Sig(output_pi).cpu()
                output_pi =ema_model.module(input.cuda())[1]["pi"]
                output_ema = Sig(output_pi).cpu()
        
        target = target.cuda(non_blocking=True)
        Normlabels = target
        if args.dataset == "COCO":
            Normlabels = (target.max(dim=1)[0]).float()
            target = Normlabels
        elif args.dataset == "PascalVOC":
            Normlabels = (target >= 0).float().cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        # for mAP calculation
        preds_regular.append(output_regular.detach().cpu())
        preds_ema.append(output_ema.detach().cpu())
        targets.append(target.detach().cpu())
    # saved data
    targets = torch.cat(targets).numpy()
    preds_regular = torch.cat(preds_regular).numpy()
    preds_ema = torch.cat(preds_ema).numpy()

    data_regular = np.concatenate((preds_regular, targets), axis=1)
    saved_name_regular = 'tmpdata/data_regular_tmp.{}.txt'.format(dist.get_rank())
    np.savetxt(os.path.join(args.output, saved_name_regular), data_regular)
    data_ema = np.concatenate((preds_ema, targets), axis=1)
    saved_name_ema = 'tmpdata/data_ema_tmp.{}.txt'.format(dist.get_rank())
    np.savetxt(os.path.join(args.output, saved_name_ema), data_ema)
    if dist.get_world_size() > 1:
        dist.barrier()

    if dist.get_rank() == 0:
        logger.info("Calculating mAP:")
        filenamelist_regular = ['tmpdata/data_regular_tmp.{}.txt'.format(ii) for ii in range(dist.get_world_size())]
        
        
        predictions, labels = pred_merger([os.path.join(args.output, _filename) for _filename in filenamelist_regular], args.num_classes)
        
        Modelmeter = on_end_batch(Modelmeter,predictions.detach().cpu(), labels.detach().cpu())
        
        test_accListModel = on_end_epoch(Modelmeter, training=False, config=args, distributed=False)
        
       
        filenamelist_ema = ['tmpdata/data_ema_tmp.{}.txt'.format(ii) for ii in range(dist.get_world_size())]
        
        predictions, labels = pred_merger([os.path.join(args.output, _filename) for _filename in filenamelist_ema], args.num_classes)
        
        Emameter =  on_end_batch(Emameter,predictions.detach().cpu(), labels.detach().cpu())
        
        test_accListEma = on_end_epoch(Emameter, training=False, config=args, distributed=False)
        
        mAP_score_regular = test_accListModel["map"]
    
        mAP_score_ema = test_accListEma["map"]
        
        for i in ["map", "OP", "OR", "OF1", "CP", "CR", "CF1", "OP_3", "OR_3", "OF1_3", "CP_3", "CR_3", "CF1_3"]:
            losses[str("model_"+i)].append(test_accListModel[i])
            losses[str("ema_"+i)].append(test_accListEma[i])
        for i in ["map"]:#, "OP", "OR", "OF1", "CP", "CR", "CF1", "OP_3", "OR_3", "OF1_3", "CP_3", "CR_3", "CF1_3"]:
            losses[str("model_"+i)].append(mAP_score_regular)
            losses[str("ema_"+i)].append(mAP_score_ema)
            
        if Modelmeter['ap_meter'].difficult_example:
            object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']
        
            if test_accListModel["map"] > test_accListEma["map"]:
                map_scores = test_accListModel["class_map"]  
            else:
                map_scores = test_accListEma["class_map"]
            
            for idx in range(len(object_categories)):
                losses[object_categories[idx]].append(map_scores[idx].item())
        
        df_losses = pd.DataFrame(data=losses)
        df_losses.to_excel("Experiments_ASL-{}_{}_ID_{}.xlsx".format(args.dataset, args.model_name, id))

        logger.info("mAP score regular {:.4f}, mAP score EMA {:.4f}".format(mAP_score_regular, mAP_score_ema))
    else:
        mAP_score_regular = 0
        mAP_score_ema = 0

    return mAP_score_regular, mAP_score_ema


if __name__ == '__main__':
    main()
