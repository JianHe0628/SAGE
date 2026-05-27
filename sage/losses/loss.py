import torch
import torch.nn as nn
import torch.nn.functional as F

class CMLoss(nn.Module):
    def __init__(self, is_similarity=False, is_symmetric=True):
        super().__init__()
        self.is_symmetric = is_symmetric
        self.is_similarity = is_similarity
    
    def forward(self, y_pred, target):
        assert len(y_pred) == len(target), "Batch size of y_pred and target should be the same"
        fn = self.similarity if self.is_similarity else self.loss
        out = fn(y_pred, target)
        if self.is_symmetric:
            out = (out + fn(target, y_pred))/2
        return out
        
    def loss(self, y_pred, target):
        batch_similarity = self.similarity(y_pred, target)
        loss = -1*batch_similarity.log()
        return loss.mean()
    
    def similarity(self, y_pred, target):
        
        max_sim = []
        for source, target in zip(y_pred, target):
            source_norm = F.normalize(source, dim=-1)
            target_norm = F.normalize(target, dim=-1)
            similarity = torch.matmul(source_norm, target_norm.T).clamp(min=1e-8)
            similarity = similarity.max(-1)[0]
            max_sim.append(similarity.mean())
        return torch.stack(max_sim).cuda()

class CMDLoss(nn.Module):
    def __init__(self, offset=1, bandwidth=0.5, is_similarity=False, is_symmetric=True):
        super().__init__()
        self.is_symmetric = is_symmetric
        self.is_similarity = is_similarity
        self.offset = offset
        self.bandwidth = bandwidth
    
    def forward(self, y_pred, target):
        assert len(y_pred) == len(target), "Batch size of y_pred and target should be the same"
        fn = self.similarity if self.is_similarity else self.loss
        out = fn(y_pred, target)
        if self.is_symmetric:
            out = (out + fn(target, y_pred))/2
        return out
        
    def loss(self, y_pred, target):
        batch_similarity, internal_similarity = self.similarity(y_pred, target)
        loss = -1*batch_similarity.log()
        return loss.mean() + internal_similarity.mean()
    
    def similarity(self, y_pred, target):
        internal_sim_list = []
        max_sim = []
        for source, target in zip(y_pred, target):
            source_norm = F.normalize(source, dim=-1)
            target_norm = F.normalize(target, dim=-1)

            internal_sim = torch.matmul(source_norm, source_norm.T)
            internal_sim = internal_sim - torch.eye(internal_sim.size(0)).cuda()
            internal_sim_list.append(internal_sim.abs().mean())
            
            # Compute squared norms
            source_norm_sqrt = (source_norm ** 2).sum(dim=-1, keepdim=True)  # (T_b, 1)
            target_norm_sqrt = (target_norm ** 2).sum(dim=-1, keepdim=True)  # (M_b, 1)

            # Compute dot product similarity
            diffs = source_norm @ target_norm.t() # (T, M)

            # Compute squared Euclidean distance: ||a - b||^2 = ||a||^2 - 2a·b + ||b||^2
            # We need to broadcast the norms properly.
            distances = source_norm_sqrt - 2 * diffs + target_norm_sqrt.t()
            distances.clamp_(min=0.0)

            min_dist = distances.min(1)[0]
            norm_dist = distances / (min_dist.unsqueeze(1) + 1e-5) 
            similarity = torch.exp((self.offset - norm_dist) / self.bandwidth)
            norm_similarity = similarity / similarity.sum(-1).unsqueeze(-1)
            norm_similarity = norm_similarity.max(-1)[0]

            max_sim.append(norm_similarity.mean())
        return torch.stack(max_sim).cuda(), torch.stack(internal_sim_list).cuda()
    

class TripletCMDLoss(nn.Module):
    def __init__(self, offset=1, bandwidth=0.5, is_similarity=True, is_symmetric=True, margin=0.5):
        super().__init__()
        self.is_symmetric = is_symmetric
        self.is_similarity = is_similarity
        self.offset = offset
        self.bandwidth = bandwidth
        self.margin = margin
        self.cx = CMDLoss(offset, bandwidth, is_similarity, is_symmetric)

    
    def forward(self, y_pred, target):
        assert len(y_pred) == len(target), "Batch size of y_pred and target should be the same"
        anchor = y_pred
        positive = target
        negative = target[::-1]
        loss = self.triplet_loss(anchor, positive, negative).mean()
        return loss
        
        
    def triplet_loss(self, anchor, positive, negative):
        pos_sim = self.cx(anchor, positive)
        neg_sim = self.cx(anchor, negative)
        loss = F.relu(neg_sim - pos_sim + self.margin)
        return loss
    
class CLCLLoss(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss