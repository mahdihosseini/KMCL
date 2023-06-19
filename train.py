import argparse

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
import os
from tqdm import tqdm
import collections
import json

from torch.optim import lr_scheduler
from src.helper_functions.helper_functions import CocoDetection, CutoutPIL, ModelEma, add_weight_decay
from src.models import create_model
from src.loss_functions.losses import KMCL_Loss
from Kmcl_Class import KMCL
from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast
from src.datasets.VOC import VOC2007
from src.datasets.ADPDataset import ADPDataset
from src.datasets.XrayDataset import CXRDataset
from src.datasets.subCOCO import CocoSubDetection
from meters import *
import pandas as pd
from pthflops import count_ops
from ptflops import get_model_complexity_info


# Standard Argument Setup
torch.multiprocessing.set_sharing_strategy('file_system')


parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('--data_folder', help='path to dataset', default='/fs2/comm/kpgrp/mhosseini/project_MCL/')
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--model-name', default='tresnet_m')
parser.add_argument('--model-path', 
                    default='/fs2/comm/kpgrp/mhosseini/github/KMCL/networks/models/tresnet_m_miil_21k.pth',
                    type=str)
parser.add_argument('--num-classes', default=80)
parser.add_argument('--epochs', default=40)
parser.add_argument('--dataset', 
                    choices=["PascalVOC", "COCO", "ADP", "Xray", "COCOSub"], default="COCOSub")
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--image-size', default=224, type=int,
                    metavar='N', help='input image size (default: 448)')

parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--print-freq', '-p', default=64, type=int,
                    metavar='N', help='print frequency (default: 64)')

parser.add_argument('--dtgfl', action='store_true', 
            help='using disable_torch_grad_focal_loss in ASL loss')
parser.add_argument('--output',
                    help='path to output folder', default="newLogs/")
parser.add_argument('--multi', action='store_true', help='using dataparallel')

parser.add_argument('--loss-opt', default="all", choices=["all", "ASLOnly"], help='loss type, only ASL vs ASL + KMCL + REC')

parser.add_argument('--loss-case', default="anisotropic", choices=["isotropic", "anisotropic"], help='loss case')

parser.add_argument('--similarity', default="BC", choices=["BC", "MKS", "RBF"], help='similarity metric')

parser.add_argument('--num-samples-sub', default=0, type=int, help='Only set for sub-sampling the Datasets, else 0')


def main():
    args = parser.parse_args()
    channels = 3
    imageSize = args.image_size
    if args.dataset == "ADP":
        num_classes = 9
        
    elif args.dataset == 'PascalVOC':
        num_classes = 20
        
    elif args.dataset == 'COCO':
        num_classes = 80
    
    elif args.dataset == 'Xray':
        num_classes = 14
        channels = 1
        
    elif args.dataset == 'COCOSub':
        num_classes = 80
        assert args.num_samples_sub != 0
    else:
        raise NotImplementedError(f"{args.dataset} is not implemented")
    
    args.num_classes = num_classes
    
    args.do_bottleneck_head = False

    id = str(np.random.randint(100000))
    path = 'config-model-{}_{}_ID_{}.json'.format(args.dataset, args.model_name, id)
    with open(path, 'w') as f:
        json.dump(vars(args), f, indent=2)
            
    # Setup model
    print('creating model...')
    encoder_model, dim = create_model(args)
    encoder_model = encoder_model.cuda()
   
    if args.model_path:  # make sure to load pretrained ImageNet model
        state = torch.load(args.model_path, map_location='cpu')
        filtered_dict = {k: v for k, v in state['state_dict'].items() if
                         (k in encoder_model.state_dict() and 'head.fc' not in k)}
        encoder_model.load_state_dict(filtered_dict, strict=False)
        
    model = KMCL(encoder_model, dim, out_classes=args.num_classes, args=args)
    
    mac, param = get_model_complexity_info(model, (channels, imageSize, imageSize), as_strings=True,
                                               print_per_layer_stat=False, verbose=True)
    print(mac, param)

    print('done\n')

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
    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    # Actuall Training
    train_multi_label(model, train_loader, val_loader, args.lr, args, id)


def train_multi_label(model, train_loader, val_loader, lr, args, id):
    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82
    
    
    losses = collections.defaultdict(list)
    # set optimizer
    Epochs = args.epochs
    weight_decay = 1e-4
    criterion = KMCL_Loss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True, loss_case=args.loss_case, loss_opt = args.loss_opt)
    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,
                                        pct_start=0.2)
    
    
    highest_mAP = 0
    trainInfoList = []
    scaler = GradScaler()
    for epoch in range(Epochs):
        LossDict = {
        "ASLLoss":[],
        "NLLLoss" :[],
        "BDLoss" :[]}
        for i, (inputData, target) in enumerate(train_loader):
                       
            inputData = inputData.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            Normlabels = target
            if args.dataset == "COCO" or args.dataset == "COCOSub":
                Normlabels = (target.max(dim=1)[0]).float()
                target = Normlabels
            elif args.dataset == "PascalVOC":
                Normlabels = (target >= 0).float().cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            
            with autocast():  # mixed precision
                
                features, gaussian_params = model(inputData)
                
            loss, ASLLoss, NLLLoss, BDLoss = criterion(features, gaussian_params, Normlabels)
            
            
            model.zero_grad()
            if torch.isfinite(loss).item():
                scaler.scale(loss).backward()
                

                scaler.step(optimizer)
                scaler.update()
                

                scheduler.step()

                ema.update(model)
                LossDict["ASLLoss"].append(ASLLoss.detach().cpu())
                LossDict["NLLLoss"].append(NLLLoss.detach().cpu())
                LossDict["BDLoss"].append(BDLoss.detach().cpu())
            # store information
            if i % 100 == 0:
                trainInfoList.append([epoch, i, loss.item()])
                print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f},  ASLLoss: {:.1f},  NLLLoss: {:.1f},  BDLoss: {:.1f}'
                      .format(epoch, Epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3),
                              scheduler.get_last_lr()[0], \
                              loss.item(),
                              ASLLoss.item(), NLLLoss.item(), BDLoss.item()))
            
        if epoch % 10 == 0 or epoch == 39:
            for i in ["ASLLoss", "BDLoss", "NLLLoss"]:
                try:
                    mean = torch.mean(torch.stack(LossDict[i])).item()
                except:
                    mean = float("nan")
            losses[i].append(mean)
            model.eval()
            mAP_score = validate_multi(val_loader, model, ema, args, losses, id)
            model.train()
            if mAP_score > highest_mAP:
                highest_mAP = mAP_score
                try:
                    torch.save(model.state_dict(), os.path.join(
                        'saved_models/', 'model-asl-{}_{}_ID_{}.ckpt'.format(args.dataset, args.model_name, id)))
                    
                    torch.save(ema.module.state_dict(), os.path.join(
                        'saved_models/', 'ema-model-asl-{}_{}_ID_{}.ckpt'.format(args.dataset, args.model_name, id)))
                    
                    
                except:
                    pass
            print('ID = {}  | current_mAP = {:.2f}, highest_mAP = {:.2f}\n'.format(id, mAP_score, highest_mAP))


def validate_multi(val_loader, model, ema_model, args, losses, id):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    
    de = False
    if args.dataset == "PascalVOC":
        de = True
    Modelmeter = initialize_meters(dist=False, difficult_example=de, ws=4)
    Modelmeter = on_start_epoch(Modelmeter)
    
    Emameter = initialize_meters(dist=False, difficult_example=de, ws=4)
    Emameter = on_start_epoch(Emameter)
    
    for i, (input, target) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            with autocast():   
                output_pi = model(input.cuda())[1]["pi"]
                output_regular = Sig(output_pi).cpu()
                output_pi =ema_model.module(input.cuda())[1]["pi"]
                output_ema = Sig(output_pi).cpu()
        
        if args.dataset == "COCO" or args.dataset == "COCOSub":
            Normlabels = (target.max(dim=1)[0]).float()
            target = Normlabels
        elif args.dataset == "PascalVOC":
            Normlabels = (target >= 0).float().cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            
        Modelmeter = on_end_batch(Modelmeter,output_regular.detach().cpu(), target.detach().cpu())
        Emameter =  on_end_batch(Emameter,output_ema.detach().cpu(), target.detach().cpu())
        
    
    test_accListModel = on_end_epoch(Modelmeter, training=False, config=args, distributed=False)
    
    test_accListEma = on_end_epoch(Emameter, training=False, config=args, distributed=False)
    
    mAP_score_regular = test_accListModel["map"]
    
    mAP_score_ema = test_accListEma["map"]
    

    for i in ["map", "OP", "OR", "OF1", "CP", "CR", "CF1", "OP_3", "OR_3", "OF1_3", "CP_3", "CR_3", "CF1_3"]:
        losses[str("model_"+i)].append(test_accListModel[i])
        losses[str("ema_"+i)].append(test_accListEma[i])
    
    # PASCAL_VOC
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
    df_losses.to_excel("Experiments_{}_{}_ID_{}.xlsx".format(args.dataset, args.model_name, id))
    
    print("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))
    return max(mAP_score_regular, mAP_score_ema)


if __name__ == '__main__':
    main()
