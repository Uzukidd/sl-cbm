import torch
import torch.nn as nn
import torch.nn.functional as F


class FARE_loss(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, input:torch.Tensor, 
                    target:torch.Tensor):
        pass
        

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
        self.scale = 1.0/1e-4
    
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