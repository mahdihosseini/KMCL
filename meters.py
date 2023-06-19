import torch
import numpy as np
import math
import torch.distributed as dist
import os

threshold = 0.8

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False

    if not dist.is_initialized():
        return False

    return True

def save_on_master(*args, **kwargs):

    if is_main_process():
        torch.save(*args, **kwargs)

def get_rank():

    if not is_dist_avail_and_initialized():
        return 0

    return dist.get_rank()

def is_main_process():

    return get_rank() == 0

class AveragePrecisionMeter(object):
    """
    The APMeter measures the average precision per class.
    The APMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
    contains model output scores for `N` examples and `K` classes that ought to
    be higher when the model is more convinced that the example should be
    positively labeled, and smaller when the model believes the example should
    be negatively labeled (for instance, the output of a sigmoid function); (2)
    the `target` contains only values 0 (for negative examples) and 1
    (for positive examples); and (3) the `weight` ( > 0) represents weight for
    each sample.
    """

    def __init__(self, dist=False, difficult_example=False, ws=4):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()
        self.distributed = dist
        self.difficult_example = difficult_example
        self.world_size = ws
    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.FloatTensor(torch.FloatStorage()).cuda()
        self.targets = torch.LongTensor(torch.LongStorage()).cuda()

    def get_results(self):
        return self.targets, self.scores
    def add(self, output, target):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

    def value(self):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """
        if self.distributed:
            ScoreSet = [torch.zeros_like(self.scores) for _ in range(self.world_size)]
            TargetSet = [torch.zeros_like(self.targets) for _ in range(self.world_size)]
            
            # print("sc1", self.scores.shape)
            dist.all_gather(ScoreSet,self.scores)
            dist.all_gather(TargetSet,self.targets)
            
            ScoreSet = torch.cat(ScoreSet)
            TargetSet = torch.cat(TargetSet)
            self.scores = ScoreSet.detach().cpu()
            self.targets = TargetSet.detach().cpu()
        # print("sc2", self.scores.shape)
        # else:
        #     return torch.tensor([0.0])
        
        # dist.all_reduce_multigpu(self.scores, 0)
        # dist.all_reduce_multigpu(self.targets, 0)
        # print("sc2", self.targets.shape)
        # print(self.t)
        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0)).float()
        # compute average precision for each class
        
        for k in range(self.scores.size(1)):
            # print("k", k)
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            # compute average precision
            # if self.difficult_example:
            ap[k] = AveragePrecisionMeter.average_precision(scores, targets,self.difficult_example)
            # else:
                # ap[k] = AveragePrecisionMeter.average_precision_coco(scores, targets)
        return ap

    @staticmethod
    def average_precision(output, target, difficult_example):

        # sort examples
        # print("fin")
        # print(output.shape)
        sorted, indices = torch.sort(output, dim=0, descending=True)
        # print(indices)
        # indices = range(len(output))
        # Computes prec@i
        # print(target)
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        for i in indices:
            
            label = target[i]
            if difficult_example and label == 0:
                continue
            if label == 1:
                pos_count += 1
            total_count += 1
            if label == 1:
                precision_at_i += pos_count / total_count
        try:
            precision_at_i /= pos_count
        except ZeroDivisionError:
            precision_at_i = 0
        return precision_at_i
    
    @staticmethod
    def average_precision_coco(output, target):
        epsilon = 1e-8
        output=output.cpu().numpy()
        # sort examples
        indices = output.argsort()[::-1]
        # Computes prec@i
        total_count_ = np.cumsum(np.ones((len(output), 1)))

        target_ = target[indices]
        ind = target_ == 1
        pos_count_ = np.cumsum(ind)
        total = pos_count_[-1]
        pos_count_[np.logical_not(ind)] = 0
        pp = pos_count_ / total_count_
        precision_at_i_ = np.sum(pp)
        precision_at_i = precision_at_i_ / (total + epsilon)

        return precision_at_i

    def overall(self):

        scores = self.scores.cpu().numpy()
        targets = self.targets.cpu().numpy()
        scoring = np.where(scores>=threshold, 1, 0)  
        if self.difficult_example:
            targets[targets == -1] = 0      
        return self.evaluation(scoring, targets)

    def overall_topk(self, k):
        # print(self.scores, self.targets)
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        n, c = self.scores.size()
        scores = np.zeros((n, c))
        index = self.scores.topk(k, 1, True, True)[1].cpu().numpy()
        # print(index)
        tmp = self.scores.cpu().numpy()
        for i in range(n):
            for ind in index[i]:
                scores[i, ind] = 1 if tmp[i, ind] >=threshold else 0 ### Thersholder!!!
        return self.evaluation(scores, targets)


    def evaluation(self, scores_, targets_):
        n, n_class = scores_.shape
        Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
        # print(scores_)
        for k in range(n_class):
            scores = scores_[:, k]
            targets = targets_[:, k]
            targets[targets == -1] = 0
            Ng[k] = np.sum(targets == 1)
            Np[k] = np.sum(scores == 1)
            Nc[k] = np.sum(targets * (scores == 1))
        # print(np.sum(Nc), np.sum(Np), np.sum(Ng))
        Np[Np == 0] = 1
        OP = np.sum(Nc) / np.sum(Np)
        OR = np.sum(Nc) / np.sum(Ng)
        OF1 = (2 * OP * OR) / (OP + OR)
        OF1 = OF1 if not math.isnan(OF1) else 0

        CP = np.sum(Nc / Np) / n_class
        CR = np.sum(Nc / Ng) / n_class
        CF1 = (2 * CP * CR) / (CP + CR)
        CF1 = CF1 if not math.isnan(CF1) else 0
        return OP, OR, OF1, CP, CR, CF1
    
def on_start_epoch(meter):
    meter['ap_meter'].reset()
    return meter

def on_end_epoch(meter, training, config, epoch=0, distributed=False):
    map = 100 * meter['ap_meter'].value()
    class_map = None
    if  meter['ap_meter'].difficult_example:
        class_map = map
    map = map.mean()
    OP, OR, OF1, CP, CR, CF1 = meter['ap_meter'].overall()
    OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = meter['ap_meter'].overall_topk(3)
    if distributed:
        local_rank = int(os.environ.get("SLURM_LOCALID")) if config.computecanada else int(os.environ['LOCAL_RANK'])
    else:
        local_rank = 0
    if not distributed or (local_rank == 0):
        if training:

            print('Epoch: [{0}]\t'
                    'mAP {map:.3f}'.format(epoch, map=map))
            print('OP: {OP:.4f}\t'
                    'OR: {OR:.4f}\t'
                    'OF1: {OF1:.4f}\t'
                    'CP: {CP:.4f}\t'
                    'CR: {CR:.4f}\t'
                    'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
        else:

            print('Test: \t mAP {map:.3f}'.format(map=map))
            print('OP: {OP:.4f}\t'
                    'OR: {OR:.4f}\t'
                    'OF1: {OF1:.4f}\t'
                    'CP: {CP:.4f}\t'
                    'CR: {CR:.4f}\t'
                    'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
            print('OP_3: {OP:.4f}\t'
                    'OR_3: {OR:.4f}\t'
                    'OF1_3: {OF1:.4f}\t'
                    'CP_3: {CP:.4f}\t'
                    'CR_3: {CR:.4f}\t'
                    'CF1_3: {CF1:.4f}'.format(OP=OP_k, OR=OR_k, OF1=OF1_k, CP=CP_k, CR=CR_k, CF1=CF1_k))
    if distributed:
        dist.barrier()
        
    return {"map": map.numpy(),"class_map":class_map, "OP": OP, "OR": OR, "OF1": OF1, "CP": CP, "CR":CR, "CF1": CF1, "OP_3": OP_k, "OR_3": OR_k, "OF1_3": OF1_k, "CP_3": CP_k, "CR_3": CR_k, "CF1_3":CF1_k} #, meter['ap_meter'].overall()
       
       
def on_end_batch(meter,preds, labels ):

    # measure mAP
    meter['ap_meter'].add(preds, labels)
    # print(preds)
    return meter
       
def initialize_meters(dist, difficult_example, ws):
    meters = {}
    meters['ap_meter'] = AveragePrecisionMeter(dist=dist, difficult_example=difficult_example, ws=ws)
    
    
    return meters




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count