import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
from captum.attr import visualization, GradientAttribution, LayerAttribution

from typing import *

from asgt import attack_utils
from .model_utils import *
from .visual_utils import *

class embedding_robust_training:
    def __init__(self, 
                 model:PCBM_Net,
                 training_forward_func:Callable,
                 classification_attak_func:Callable,
                 embedding_attak_func:Callable,
                 explain_func:Callable,
                 regularization:str,
                 targeted_concept_idx:Union[Dict[int, int], torch.Tensor],
                 k:int,
                 lam:float,
                 feature_range:Union[list, tuple],
                 device:torch.device, 
                 preprocess:Optional[transforms.Compose] = None) -> None:

        self.model = model
        
        self.training_forward_func = training_forward_func
        self.classification_attak_func = classification_attak_func
        self.embedding_attak_func = embedding_attak_func
        
        self.explain_func = explain_func
        self.preprocess = preprocess
        
        self.targeted_concept_idx = targeted_concept_idx

        self.k = k
        self.lam = lam
        self.feature_range = feature_range
        self.device = device
        
        if isinstance(self.targeted_concept_idx, dict):
            max_key = max(targeted_concept_idx.keys())
            max_indices = list(targeted_concept_idx.values())[0].size(0)
            mapping_tensor = torch.empty((max_key + 1, max_indices), dtype=torch.int64).to(device)
            for k, v in targeted_concept_idx.items():
                mapping_tensor[k] = v
            
            self.targeted_concept_idx = mapping_tensor
            
        self.regularization_loss_func:Optional[Callable[..., torch.Tensor]] = None
        if regularization is not None:
            self.regularization_loss_func = getattr(self, f"_regular_func_{regularization}")
        
        print(self.targeted_concept_idx)
    
    def _regular_func_concept_KL_div(self, clean_embedding:torch.Tensor,
                                     adv_embedding:torch.Tensor,
                                     masked_adv_embedding:torch.Tensor):
        
        clean_concept_proj = self.model.comput_dist(clean_embedding)        
        masked_adv_concept_proj = self.model.comput_dist(masked_adv_embedding)
        
        return nn.functional.kl_div(clean_concept_proj.log_softmax(dim=1), 
                                                  masked_adv_concept_proj.log_softmax(dim=1), 
                                                  reduction='batchmean', 
                                                  log_target=True)
        
    def _iterate(self, batch_X:torch.Tensor,
                      batch_Y:torch.Tensor):
        B, C, W, H = batch_X.size()
        self.model.output_type("embedding")
        self.model.eval()
        batch_Y_concepts = self.targeted_concept_idx[batch_Y]
        original_embedding = self.model.embed(batch_X).detach()
        batch_adv_X:torch.Tensor = self.embedding_attak_func(batch_X, original_embedding)
        
        self.model.output_type("concepts")
        
        expanded_batch_X = batch_X.unsqueeze(1).expand(-1, batch_Y_concepts.size(1), -1, -1, -1)
        expanded_batch_X = expanded_batch_X.reshape(-1, C, W, H)
        viewed_batch_Y_concepts = batch_Y_concepts.view(-1)

        adv_attributions = self.explain_func(expanded_batch_X, target = viewed_batch_Y_concepts)
        batch_masked_adv_x = self.generate_masked_sample(expanded_batch_X, adv_attributions)
        

        self.model.output_type("embedding")
        self.model.train()
        clean_embedding = self.model(batch_X)
        adv_embedding = self.model(batch_adv_X)
        masked_adv_embedding = self.model(batch_masked_adv_x)
        
        clean_concept_proj = self.model.comput_dist(clean_embedding)
        clean_logit = self.model.forward_projs(clean_concept_proj)

        expanded_clean_embedding = clean_embedding.unsqueeze(1).expand(-1, batch_Y_concepts.size(1), -1).reshape(B * batch_Y_concepts.size(1), -1)
        
        loss = nn.functional.cross_entropy(clean_logit, batch_Y) \
            + nn.functional.mse_loss(clean_embedding, 
                                      adv_embedding) \
            + self.lam * self.regularization_loss_func(expanded_clean_embedding, 
                                                       adv_embedding,
                                                       masked_adv_embedding)

        if self.training_forward_func is not None:
            self.training_forward_func(loss)
        
        return loss.item()
    
    def evaluate_embedding_robustness(self, data_loader):
        self.model.output_type("embedding")
        self.model.eval()
        
        totall_accuracy = []
        totall_l2_loss = []
        for idx, data in tqdm(enumerate(data_loader), 
                            total=data_loader.__len__()):
            batch_X, batch_Y = data
            batch_X:torch.Tensor = batch_X.to(self.device)
            batch_Y:torch.Tensor = batch_Y.to(self.device)

            original_embedding = self.model.embed(batch_X).detach()

            batch_adv_X = self.embedding_attak_func(batch_X, original_embedding)
            adv_embedding = self.model(batch_adv_X)
            
            adv_logit = self.model.forward_projs(self.model.comput_dist(adv_embedding))
        
            loss = nn.functional.mse_loss(original_embedding, adv_embedding)
            totall_l2_loss.append(loss.item())
            totall_accuracy.append((adv_logit.argmax(1) == batch_Y).float().mean().item())
            
        print(f"Embedding l2 loss: {np.array(totall_l2_loss).mean()}")
        print(f"Embedding robustness accuracy: {np.array(totall_accuracy).mean()}")
        
    def evaluate_model(self, data_loader):
        self.model.output_type("logit")
        return attack_utils.evaluate_model(self.model, 
                                           data_loader,
                                           self.device)
        
    def evaluate_model_robustness(self, data_loader):
        self.model.output_type("logit")
        return attack_utils.evaluate_model_robustness(self.model,
                                               data_loader,
                                               self.classification_attak_func,
                                               self.device)
        
    def generate_masked_sample(self, batch_X:torch.Tensor,
                        attributions:torch.Tensor):
        
        B, C, W, H = batch_X.size()
        self.model.eval()
        
        attributions = attributions.mean(1).abs().view(B, W * H)
        _, attributions_masked_indices = torch.topk(attributions, self.k, dim=1,largest=False)
        attributions_masked_indices = attributions_masked_indices.unsqueeze(1).expand(-1, C, -1)
        
        _random_values = attributions.new_empty(attributions_masked_indices.size()).uniform_(*self.feature_range)
        masked_batch_X = batch_X.detach().clone().view(B, C, -1)
        masked_batch_X.scatter_(2, attributions_masked_indices, _random_values)
        masked_batch_X = masked_batch_X.view(B, C, W, H)
        
        return masked_batch_X
        
    def generate_adv_sample(self, batch_X:torch.Tensor,
                                    batch_Y:torch.Tensor):
        B, C, W, H = batch_X.size()
        self.model.eval()
        
        batch_adv_X:torch.Tensor = self.embedding_attak_func(batch_X, 
                                                    batch_Y)
        
        return batch_adv_X
    
    @staticmethod
    def show_image(images:torch.Tensor, comparison_images:torch.Tensor=None):
        import torchvision
        import matplotlib.pyplot as plt
        
        if comparison_images is not None:
            images = torch.cat((images, comparison_images), dim=3)
        
        images = images.detach().cpu()
        grid_img = torchvision.utils.make_grid(images, nrow=2, normalize=True)

        plt.imshow(grid_img.permute(1, 2, 0))
        plt.axis('off')
        plt.show()
        import pdb; pdb.set_trace()
        
    def train_one_epoch(self, data_loader:DataLoader, use_tqdm=True):
        running_loss = 0

        if use_tqdm:
            data_loader = tqdm(data_loader, 
                            total=data_loader.__len__())

        for data in data_loader:
            batch_X, batch_Y = data
            batch_X:torch.Tensor = batch_X.to(self.device)
            batch_Y:torch.Tensor = batch_Y.to(self.device)

            # self.show_image(batch_X, batch_adv_X)
            loss = self._iterate(batch_X, 
                                batch_Y)
            running_loss += loss
            
            
        return running_loss