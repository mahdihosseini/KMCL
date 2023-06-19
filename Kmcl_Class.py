import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module as Module
import time
class KMM(Module):
    r'''
    Implementation of the Gaussian Mixture Model
    seen in Multi-label Contrastive Learning.

    Args:
        in_features (int): number of input features
        out_classes (int): number of output classes
            for ADP, 33 classes

    Forward call:
        returns (dict):
            {
            "pi" : pi_out,
            "mean" : mean_out,
            "var" : var_out
            }
    '''

    def __init__(self, in_features: int, out_classes: int, args=None, batch=None):
        super(KMM, self).__init__()
        self.in_features = in_features
        self.out_classes = out_classes
        self.sim = args.similarity
        self.num_classes = out_classes
        self.pi = nn.Linear(in_features, out_classes) #nn.Parameter(torch.Tensor(out_classes, in_features))
        
        if args.loss_opt == 'ASLOnly': 
            self.loss = args.loss_opt
            return
        self.loss = args.loss_opt
        
        if args.loss_case == "isotropic":
            self.mean = nn.Linear(in_features, out_classes)
            self.var = nn.Linear(in_features, out_classes)
        elif args.loss_case == "anisotropic":
            self.mean = nn.Linear(in_features, out_classes*in_features)
            self.var = nn.Linear(in_features, out_classes*in_features)
            
            
        if self.sim == "MKS":
            self.var = nn.Linear(in_features, in_features)
        
        elif self.sim == "RBF":
            self.var = nn.Linear(in_features, 1)
        
        
      
        self.reset_parameters(batch)  # init params

    def reset_parameters(self, batch=None):
        # TODO tweak this since params and style probably doesn't work optimally

       
       
        nn.init.uniform_(self.mean.weight, a=0.0, b=0.1)
        nn.init.ones_(self.var.weight)

       
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.var.bias)
        return
             
    def forward(self, inp):
        # ''' input must be (num_samples * ... * in_features)

        # returns:
        #     pi : Tensor of shape (num_samples * ... * out_classes)
        #     mean : mean_out (num_samples * ... * out_classes)
        #     var : var_out (num_samples * ... * out_classes)

        # '''
        # s = time.time()
        pi_a = self.pi(inp)#torch.tensordot(inp, self.pi_weight, dims=([-1], [1])) + self.pi_bias
        pi_out = pi_a #F.sigmoid(pi_a)
        if self.loss != 'ASLOnly': 
            mean_a = self.mean(inp)#torch.tensordot(inp, self.mean_weight, dims=([-1], [1])) + self.mean_bias
            var_a = self.var(inp)#torch.tensordot(inp, self.var_weight, dims=([-1], [1])) + self.var_bias
            
            
            mean_out = mean_a#torch.tensor(0, dtype=torch.float)#mean_a
            
            var_out = 1e-7 + 1 + 1 + nn.ELU()(var_a)#torch.tensor(0, dtype=torch.float)#
            if self.sim == "MKS":
                var_out = var_out.repeat_interleave(self.num_classes)
                
            elif self.sim == "RBF": # and Isotropic
                var_out = (var_out.repeat_interleave(self.num_classes))#.repeat_interleave(self.in_features)
                var_out = var_out.reshape((-1, self.num_classes))
        else:
            mean_out = None
            var_out = None
        return {
            "pi": pi_out,
            "mu": mean_out,
            "var": var_out
        }
        # return {"pi": pi_out}
    # def __repr__(self) -> str:
    #     return self.extra_repr()



class KMCL(Module):
    def __init__(self, model, dim, out_classes, args):
        super(KMCL, self).__init__()
        
        self.encoder = model
        
        self.loss_opt = args.loss_opt
        if args.loss_case == "anisotropic":
            newDim = 128
            self.downsampler = nn.Conv1d(dim, newDim,1).cuda()
            dim = newDim
            self.loss_case = "anisotropic"
        else:
            self.loss_case = "isotropic"
        self.kmm = KMM(in_features=dim, out_classes=out_classes, args=args).cuda()
    
    def forward(self, x):
        # start = time.time()
        feat = self.encoder(x)
        # print(time.time() - start)
        if self.loss_case == "anisotropic":
            feat = torch.squeeze(self.downsampler(torch.unsqueeze(feat,-1)))
        params = self.kmm(feat)
        # 
        if self.loss_opt == "ASLOnly":
            return None, params
        
        # feat = F.normalize(feat, dim=1)
        return feat, params   
    
    def non_norm_forward(self, x):
        
        feat = self.encoder(x)
        if self.loss_case == "anisotropic":
            feat = torch.squeeze(self.downsampler(torch.unsqueeze(feat,-1)))
        params = self.kmm(feat)
        # feat = F.normalize(feat, dim=1)
        return feat, params   