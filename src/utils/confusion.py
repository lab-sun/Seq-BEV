import torch
import numpy as np

class BinaryConfusionMatrix(object):

    def __init__(self, num_class):
        self.tp = torch.zeros(num_class, dtype=torch.long)
        self.fp = torch.zeros(num_class, dtype=torch.long)
        self.fn = torch.zeros(num_class, dtype=torch.long)
        self.tn = torch.zeros(num_class, dtype=torch.long)


    @property
    def num_class(self):
        return len(self.tp)
    
    def update(self, preds, labels, mask=None):

        preds = preds.detach().cpu()
        labels = labels.detach().cpu()

        # Move batch dimension to the end
        preds = preds.flatten(2, -1).permute(1, 0, 2).reshape(
            self.num_class, -1)
        labels = labels.flatten(2, -1).permute(1, 0, 2).reshape(
            self.num_class, -1)

        if mask is not None:
            preds = preds[:, mask.flatten()]
            labels = labels[:, mask.flatten()]
        

        true_pos = preds & labels
        false_pos = preds & ~labels
        false_neg = ~preds & labels
        true_neg = ~preds & ~labels

        # Update global counts
        self.tp += true_pos.long().sum(-1)
        self.fp += false_pos.long().sum(-1)
        self.fn += false_neg.long().sum(-1)
        self.tn += true_neg.long().sum(-1)
    

    @property
    def iou(self):
        return self.tp.float() / (self.tp + self.fn + self.fp).float()
    
    @property
    def mean_iou(self):
        # Only compute mean over classes with at least one ground truth
        valid = (self.tp + self.fn) > 0
        if not valid.any():
            return 0
        return float(self.iou[valid].mean())

    @property
    def dice(self):
        return 2 * self.tp.float() / (2 * self.tp + self.fp + self.fn).float()
    
    @property
    def macro_dice(self):
        valid = (self.tp + self.fn) > 0
        if not valid.any():
            return 0
        return float(self.dice[valid].mean())
    
    @property
    def precision(self):
        return self.tp.float() / (self.tp + self.fp).float()
    
    @property
    def recall(self):
        return self.tp.float() / (self.tp + self.fn).float()

############################################################################################################
# sun method to calcultate IoU
def compute_results(conf_total):
    n_class =  conf_total.shape[0]
    # TODO 检查ignore_index=0时，是否将consider_unlabeled设为False
    consider_unlabeled = True  # must consider the unlabeled, please set it to True 
    if consider_unlabeled is True:
        start_index = 0
    else:
        start_index = 1
    precision_per_class = np.zeros(n_class)
    recall_per_class = np.zeros(n_class)
    iou_per_class = np.zeros(n_class)
    for cid in range(start_index, n_class): # cid: class id
        if conf_total[start_index:, cid].sum() == 0:
            #precision_per_class[cid] =  np.nan
            precision_per_class[cid] =  0.0  # 原来是nan，一起算mean时，结果总为nan，GS改为0
        else:
            precision_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[start_index:, cid].sum()) # precision = TP/TP+FP
        if conf_total[cid, start_index:].sum() == 0:
            recall_per_class[cid] = np.nan
        else:
            recall_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[cid, start_index:].sum()) # recall = TP/TP+FN
        if (conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid]) == 0:
            iou_per_class[cid] = np.nan
        else:
            iou_per_class[cid] = float(conf_total[cid, cid]) / float((conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid])) # IoU = TP/TP+FP+FN

    return precision_per_class, recall_per_class, iou_per_class