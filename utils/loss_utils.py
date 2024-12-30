import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable


class FARE_loss(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, input:torch.Tensor, 
                    target:torch.Tensor):
        pass

class probability_entropy_loss(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, input:torch.Tensor, softmax_space:bool=True):
        """
            input: [B, N, D]
        """
        if not softmax_space:
            input = F.softmax(input, dim = 2)
        
        entropy = -torch.sum(input * torch.log(input + 1e-10), dim=(1, 2)).mean()

        return entropy


"""
###############################################################################################################
                                                    Loss Functions of CSS-CBM
###############################################################################################################
"""

class ClipLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.logit_scale = 1.0/temperature

    def get_ground_truth(self, device, num_logits):
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        return labels

    def get_logits(self, img1_features, img2_features):
        logits_per_image = self.logit_scale * img1_features @ img2_features.T
        return logits_per_image

    def forward(self, x):
        img1_features, img2_features = x[:,0,:], x[:,1,:]

        img1_features = F.normalize(img1_features, dim=-1)
        img2_features = F.normalize(img2_features, dim=-1)
        
        logits_per_image = self.get_logits(img1_features, img2_features)
        labels = self.get_ground_truth(img1_features.device, logits_per_image.shape[0])
        return F.cross_entropy(logits_per_image, labels)


class ss_concept_loss(nn.Module):
    def __init__(self):
        super(ss_concept_loss, self).__init__()

        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()
        self.scale = 1.0/1e-0
    
    def forward(self, predicted_concepts, class_logits, class_label, concept_label, use_concept_labels):

        classification_loss = self.ce_loss(class_logits, class_label)

        normalized_predicted_concepts = self.scale * predicted_concepts * use_concept_labels.unsqueeze(-1).expand(-1,18)
        normalized_concept_labels = self.scale * concept_label * use_concept_labels.unsqueeze(-1).expand(-1,18)
        concept_loss = self.l1_loss(normalized_predicted_concepts, normalized_concept_labels)

        return classification_loss, concept_loss


# Trinity loss (combination of 3 loss functions) = contrastive loss + classification loss (supervised) + concept loss (semi-supervised)
class trinity_loss(nn.Module):
    def __init__(self):
        super(trinity_loss, self).__init__()

        self.clip_loss = ClipLoss()
        self.ss_loss = ss_concept_loss()

    def forward(self, predicted_concepts, class_logits, class_label, concept_label, use_concept_labels):
        bs_2, dim = predicted_concepts.shape
        contrasive_loss = self.clip_loss(torch.reshape(predicted_concepts,(int(bs_2/2),2,dim)))
        classification_loss, concept_loss = self.ss_loss(predicted_concepts, class_logits, class_label, concept_label, use_concept_labels)
        return contrasive_loss, classification_loss, concept_loss

class spss_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()
        self.pe_loss = probability_entropy_loss()
        self.scale = 1.0/1e-4

    def forward(self, predicted_concepts, class_logits, class_label, concept_label, token_concepts):
        classification_loss = self.ce_loss(class_logits, class_label)

        normalized_predicted_concepts = self.scale * predicted_concepts
        normalized_concept_labels = self.scale * concept_label
        concept_loss = self.l1_loss(normalized_predicted_concepts, normalized_concept_labels)
        entropy_loss = self.pe_loss(token_concepts, False)

        return classification_loss, concept_loss, entropy_loss
    
class ls_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()
        self.pe_loss = probability_entropy_loss()
        self.scale = 1.0/1e-3

    def forward(self, predicted_concepts, class_logits, class_label, concept_label, token_concepts):
        classification_loss = self.ce_loss(class_logits, class_label)

        normalized_predicted_concepts = self.scale * predicted_concepts
        normalized_concept_labels = self.scale * concept_label
        concept_loss = self.l1_loss(normalized_predicted_concepts, normalized_concept_labels)
        entropy_loss = self.pe_loss(token_concepts, False)

        return classification_loss, concept_loss, entropy_loss
    
class cross_entropy_concept_loss(nn.Module):
    def __init__(self, model:nn.Module,
                 explain_forward:Callable[..., torch.Tensor], 
                 feature_range:tuple[float], 
                 scale:float,
                 K:int):
        super().__init__()

        self.model= model
        self.K = K
        self.ce_loss = nn.CrossEntropyLoss()
        self.feature_range = feature_range
        self.explain_forward = explain_forward
        self.scale = scale

    def attribution_masking(self, batch_X:torch.Tensor, 
                            attributions:torch.Tensor):
        """
        batch_X : [B, C, W, H]
        attrbution : [B, F, W, H]
        """
        B, C, W, H = batch_X.size()
        attributions = attributions.mean(1).abs().view(B, W * H)
        _, attributions_masked_indices = torch.topk(attributions, self.K, dim=1, largest=False)
        attributions_masked_indices = attributions_masked_indices.unsqueeze(1).expand(-1, C, -1)
        
        _random_values = attributions.new_empty(attributions_masked_indices.size()).uniform_(*self.feature_range)
        masked_batch_X = batch_X.detach().clone().view(B, C, -1)
        masked_batch_X.scatter_(2, attributions_masked_indices, _random_values)
        masked_batch_X = masked_batch_X.view(B, C, W, H)

        return masked_batch_X
    
    def forward(self, batch_X:torch.Tensor, 
                gt_concepts:torch.Tensor,
                use_concept_labels:torch.Tensor=None,
                forward_func:Callable[..., torch.Tensor]=None):
        """
        batch_X : [B, C, W, H]
        gt_concepts  : [B, C']
        """
        if batch_X.size().__len__() == 5:
            batch_X = batch_X.view(-1, batch_X.size(-3), batch_X.size(-2), batch_X.size(-1))
        
        if use_concept_labels is not None:
            batch_X = batch_X[use_concept_labels.bool()]
            gt_concepts = gt_concepts[use_concept_labels.bool()]
        
        B, C, W, H = batch_X.size()
        batch_masked_X = []
        batch_concepts = []
        for ind_mask in range(B):
            
            ind_X = batch_X[ind_mask:ind_mask+1]
            ind_concepts = gt_concepts[ind_mask].nonzero().squeeze(1)
            if ind_concepts.numel() == 0:
                continue
            
            orginal_mode = self.model.training
            self.model.eval()
            attributions = self.explain_forward(batch_X = ind_X, target = ind_concepts)
            self.model.train(orginal_mode)

            masked_X = self.attribution_masking(ind_X.expand(ind_concepts.size(0), -1, -1, -1), attributions)
            batch_masked_X.append(masked_X)
            batch_concepts.append(ind_concepts)

        if batch_masked_X.__len__() == 0:
            return None, None
        
        batch_masked_X = torch.concat(batch_masked_X, dim = 0)
        batch_concepts = torch.concat(batch_concepts, dim = 0)

        if forward_func is not None:
            predicted_concepts = forward_func(batch_masked_X)
            loss = self.scale * self.ce_loss(predicted_concepts, batch_concepts)
            return loss, None

        return batch_masked_X, batch_concepts

        
 
        
            
            
            
