import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F


def gen_gaussian(inp, pi, mean, sd, loss_case):
    
    # Calculates the gaussian distribution at an input for a given mean and std_deviation

    # inp: Tensor of size (num_samples, in_features)
    #     feature tensor from the encoder network
    # pi: Tensor of size (num_samples, out_classes)
    # mean: Tensor of size (num_samples, out_classes)
    # sd: Tensor of size (num_samples, out_classes)
    # loss_case: "isotropic" or "anisotropic"
    # returns: Tensor of size (num_samples, out_classes)
    
        
    eps = math.sqrt(7.0/3 - 4.0/3 -1)
    sd = sd + eps
    if loss_case == "isotropic":
        exp_term = (-1 / 2) * torch.sum(
            (inp.unsqueeze(dim=-1) - mean.unsqueeze(dim=-2))**2 / (sd.unsqueeze(dim=-2))**2,
            dim=-2)
    elif loss_case == "anisotropic":
        reshaped_mean = torch.reshape(mean, (-1, inp.shape[-1], pi.shape[-1])) # [Batch x Dim x K]
        reshaped_sd = torch.reshape(sd, (-1, inp.shape[-1], pi.shape[-1])) # [Batch x Dim x K]
        exp_term = (-1 / 2) * torch.sum(
            (inp.unsqueeze(dim=-1) - reshaped_mean)**2 / (reshaped_sd)**2,
            dim=-2)
    result = torch.exp(exp_term)
    return result
    
    
class KMCL_Loss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-7, disable_torch_grad_focal_loss=False, loss_case="isotropic", loss_opt="all"):
        super(KMCL_Loss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.temp = 0.17
        self.loss_case = loss_case
        self.loss = loss_opt

    def forward(self,features, model_output, y):
        """"
        Parameters
        ----------
        features: features
        x: The output of the model {pi: input logits, mu: mean, var:variance} #input logits
        y: targets (multi-label binarized vector)
        """
        x = model_output["pi"]
        
        """ASL Loss Start"""
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        
        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w
        ASLLoss = -loss.sum()
        if self.loss == "ASLOnly":
            loss = ASLLoss #+ NLLLoss + BDLoss
            return loss, ASLLoss, torch.tensor(0), torch.tensor(0)
        
        """ASL Loss Complete"""
        
        mu = model_output["mu"]
        var = model_output["var"]
        batch_size, m = features.shape
        
        
        """Reconstruction Loss Start"""
        norm_Gaussian = gen_gaussian(features, x_sigmoid, mu, torch.sqrt(var), self.loss_case) 
        t=1.17549e-38
        norm_Gaussian2 = norm_Gaussian.clone()
        clipped_norm_gaussian = torch.clamp(norm_Gaussian2, min=t)
       
        Sim_Measure = y * x_sigmoid
        Sim_Measure = torch.where(torch.isnan(Sim_Measure), torch.zeros_like(Sim_Measure), Sim_Measure)
        
        InnerLog = (Sim_Measure * clipped_norm_gaussian).sum(1)       
        NewInnerLog = InnerLog / ((x_sigmoid *clipped_norm_gaussian).sum(1))
        
        Logged_Classes = -1*torch.log(NewInnerLog.clamp(min=self.eps))
        NLLLoss =(1 / batch_size)*(Logged_Classes.sum(0)) 
        """Reconstruction Loss Complete"""
        
        """KMCL Loss Start"""
        
        BDLoss = self.gen_bd_neurips(mu,torch.sqrt(var), m, y.float(), self.loss_case) 
        
        """KMCL Loss Complete"""
       
        loss = ASLLoss + NLLLoss + BDLoss
        return loss, ASLLoss, NLLLoss, BDLoss
    
    def gen_bd_neurips(self, mean,sd,m, labels, loss_case): #NBC
        # Calculates the Bhattacharyya Dissimilarity
        # mean: Tensor of size (num_samples, out_classes)
        # sd: Tensor of size (num_samples, out_classes)
        # returns: Tensor of size (num_samples, num_samples, out_classes, out_classes)
        batch_size,class_num = labels.shape
        eps = math.sqrt(7.0/3 - 4.0/3 -1)
        sd = sd + eps
        
        M = m
        
        if loss_case == "isotropic":
            # first term, denominator matrix      
            deno_matrix = (sd.unsqueeze(dim=1) * sd.unsqueeze(dim=0)).float()
            # denominator cannot be zero
            deno_matrix = torch.where(deno_matrix!=0,1/(2*deno_matrix),deno_matrix)
            # first term, numerator matrix
            nu_matrix = ((sd**2).unsqueeze(dim=1) + (sd**2).unsqueeze(dim=0)).float()
            # the first natural log term
            first_matrix = M * 0.5 * torch.log(nu_matrix*deno_matrix)
            # second term, numerator matrix
            mu_matrix = (mean.unsqueeze(dim=1) - mean.unsqueeze(dim=0)).float()
            mu_matrix = mu_matrix**2
            second_matrix = M * 0.25 * mu_matrix/nu_matrix
        elif loss_case == "anisotropic":
            reshaped_mean = torch.reshape(mean, (-1, M, labels.shape[-1])) # [Batch x Dim x K]
            reshaped_sd = torch.reshape(sd, (-1, M, labels.shape[-1])) # [Batch x Dim x K]
            deno_matrix = (reshaped_sd.unsqueeze(dim=0) * reshaped_sd.unsqueeze(dim=1)).float()
            # denominator cannot be zero
            deno_matrix = torch.where(deno_matrix!=0,1/(2*deno_matrix),deno_matrix)
            # first term, numerator matrix
            nu_matrix = ((reshaped_sd**2).unsqueeze(dim=0) + (reshaped_sd**2).unsqueeze(dim=1)).float()
            # the first natural log term
            first_matrix = (M * 0.5 * torch.log(nu_matrix*deno_matrix)).sum(2)
            # second term, numerator matrix
            mu_matrix = (reshaped_mean.unsqueeze(dim=0) - reshaped_mean.unsqueeze(dim=1)).float()
            mu_matrix = mu_matrix**2
            second_matrix = (M * 0.25 * mu_matrix/nu_matrix).sum(2)
           
       
        # the second term
        result = first_matrix + second_matrix
        t=1.17549e-38
        t_plus = torch.tensor(1.17549e+38)
        BC_result = torch.exp(-result)
        exp_result = torch.exp(BC_result/self.temp)
        exp_result = torch.where(torch.isinf(exp_result),t_plus.float().cuda(),exp_result)
        
        
        ## The above computes standard BC
        
        labels_1 = labels.unsqueeze(dim=1)
        labels_2 = labels.unsqueeze(dim=0)
        Mt = (labels_1 * labels_2)
        Mt_Cloned = Mt.clone().float()
        Mt_Cloned[range(batch_size), range(batch_size)] = torch.zeros(class_num).cuda()
        NewResult_Cloned = exp_result.clone()
        NewResult_Cloned[NewResult_Cloned < t] = t
        DenomResult_Cloned = exp_result.clone() # (B x B/n x K)
        
        DenomResult_Cloned[range(batch_size), range(batch_size)] = torch.zeros(class_num).cuda()
        Log_denom = DenomResult_Cloned.sum(1) #(B x K)
        Sim = NewResult_Cloned * Mt_Cloned #(B x B/n x K) is zero when B=n and when K_n != K_i
        Log_num = [i for i in Sim] #(B x B/n x K)
        Batch_entries = []
        torch.set_printoptions(threshold=10_000)
        for idx in range(len(Log_num)):
            
            # Calculate row-wise sum
            row_sum = torch.sum(Log_num[idx], dim=-1)

            # Find indices of rows where sum is not 0 or NaN
            nonzero_row_indices = torch.where((row_sum != 0) & (~torch.isnan(row_sum)))[0]
            
            Temp = Log_num[idx][nonzero_row_indices] # (B/n x K)[(B/n !=0)] --> (|A(n)| x K)
            # Check that there does exist a shared label k between anchor = i and j for j in Batch
            # Essentially the images with at least 1 shared label
            # Temp is (|A(n)| x K)
            # for anchor image idx, we have found all images with at least one matching label:
            
            
            if(Temp.shape[0] != 0 and Temp.shape[0] != len(Log_num)): # Check that the A(n) set exists ie that there are shared image labels in the batch given anchor
                Inner_division = (Temp/Log_denom[idx]).float() # (|A(n)| x K) / (K) --> (|A(n)| x K)
                # Now we have a |A(k,n)| x K matrix but we only care about the K that is not zero!
                # We take the log and still have |A(n)| x K matrix
                #Computing Cardinality: (BxK) * (Kx1) = B --> Cardinality of Union
               
                
                Jaccard_Numerator = torch.matmul(labels, labels[idx])
                Jaccard_Denominator = labels + labels[idx].repeat(len(labels),1)
                Jaccard_Denominator[Jaccard_Denominator > 1.0] = 1.0
                Jaccard_Index = Jaccard_Numerator / Jaccard_Denominator.sum(1)
                
                Jaccard_Index[idx] = 0.0 # remove anchor
                Jaccard_Index_wo_zeroes = Jaccard_Index[Jaccard_Index!=0.0]
                
                InnerArgument = torch.mul(Inner_division, Jaccard_Index_wo_zeroes[:, None]).float()
                
                InnerLog_K_Classes = torch.where(InnerArgument > 0.0, torch.log((InnerArgument+1e-7).float()).float(), torch.tensor(0, device='cuda',dtype=InnerArgument.dtype))
                # Dimension should be (A(n) x K)
                
                assert torch.isfinite(InnerLog_K_Classes).any()
                
                K_sum = InnerLog_K_Classes.sum(1).sum(0) # Should be Simply A(n) then Scalar now
        
                Cardinality_An = Temp.shape[0]
                
                
                ####### Type 1 Normalization BD
                KN_Normalized = -1*K_sum / Cardinality_An
                
                
                Batch_entries.append(KN_Normalized)
        if len(Batch_entries) == 0:
            MPCL = torch.tensor(0)
        else:
            MPCL = torch.stack(Batch_entries).sum(0)/batch_size    
        return MPCL
