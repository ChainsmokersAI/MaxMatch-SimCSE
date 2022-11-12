import torch
import torch.nn as nn
import torch.distributed as dist

class SimCSE(nn.Module):
    """
    Unsupervised SimCSE
    paper: SimCSE: Simple Contrastive Learning of Sentence Embeddings
    arXiv: https://arxiv.org/abs/2104.08821
    """
    def __init__(self, pretrained):
        super().__init__()
        
        # Pre-Trained LM
        self.pretrained=pretrained
        # Pooling Layer: MLP (Train Only)
        self.mlp=nn.Linear(self.pretrained.config.hidden_size, self.pretrained.config.hidden_size)
        
        # Cosine Similarity
        self.cos_sim=nn.CosineSimilarity(dim=-1)
        # Temperature (Hyperparam)
        self.temp=0.05
        
        # Contrastive Loss
        self.loss=nn.CrossEntropyLoss()
        
    def pooler(self, x):
        # [CLS] with MLP (Train Only)
        x=x.last_hidden_state[:,0,:]
        return self.mlp(x)
    
    def get_embedding(self, x):
        # Return Sentence Representation
        x=self.pretrained(x)
        return x.last_hidden_state[:,0,:]
    
    def forward(self, sent, pos):
        # Forward
        sent=self.pretrained(sent)
        pos=self.pretrained(pos)
        
        # Pooling
        # Shape: batch_size x hidden_dim
        repr_sent=self.pooler(sent)
        repr_pos=self.pooler(pos)

        # Multi-GPU
        if dist.is_initialized():
            repr_list_sent=[torch.zeros_like(repr_sent) for _ in range(dist.get_world_size())]
            repr_list_pos=[torch.zeros_like(repr_pos) for _ in range(dist.get_world_size())]

            # All Gather
            dist.all_gather(tensor_list=repr_list_sent, tensor=repr_sent.contiguous())
            dist.all_gather(tensor_list=repr_list_pos, tensor=repr_pos.contiguous())

            # Grad Fn
            repr_list_sent[dist.get_rank()]=repr_sent
            repr_list_pos[dist.get_rank()]=repr_pos
            
            # Shape: (world_size * batch_size) x hidden_dim
            repr_sent=torch.cat(repr_list_sent, dim=0)
            repr_pos=torch.cat(repr_list_pos, dim=0)

        # Cosine Similarity
        sim=self.cos_sim(repr_sent.unsqueeze(1), repr_pos.unsqueeze(0))/self.temp
        
        # Contrastive Loss
        label=torch.arange(sim.size(0)).long().to(sim.device)
        loss=self.loss(sim, label)
        
        return loss
