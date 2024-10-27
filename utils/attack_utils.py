from .common_utils import *
from abc import ABC, abstractmethod

@dataclass
class adversarial_attack_context:
    index:int
    original_X:torch.Tensor
    adversarial_loss_handle:torch.Tensor
    input_context:model_result
    model_context:model_pipeline
    
class adversarial_attack_funcs:
    
    @staticmethod
    def FGSM(attack_context:adversarial_attack_context, 
                eps:float):
        batch_X = attack_context.input_context.batch_X
        attack_context.adversarial_loss_handle.mean().backward()

        return (batch_X - eps * batch_X.grad.sign()).clamp_(0.0, 1.0).detach(), True
    
    @staticmethod
    def iFGSM(attack_context:adversarial_attack_context, 
                eps:float,
                max_iter:int = 5):
        endflag = attack_context.index >= max_iter
        
        alpha = eps/max_iter
        batch_X = attack_context.input_context.batch_X
        attack_context.adversarial_loss_handle.mean().backward()
        
        if torch.norm((attack_context.original_X - batch_X), 
                      p=float('inf')).item() > eps:
            endflag = True
            
        if not endflag:
            batch_X = (batch_X - alpha * batch_X.grad.sign()).clamp_(0.0, 1.0)
            
        return batch_X.detach(), endflag
    
    @staticmethod
    def PGD(attack_context:adversarial_attack_context, 
                eps:float = 0.01,
                i:int = 10):
        raise NotImplementedError

class regularization_loss:
    
    @staticmethod
    def l2_norm():
        pass

@dataclass
class ori_adv_pair:
    original_X:torch.Tensor
    adversarial_X:torch.Tensor

def evaluate_adversarial_sample(ori_adv_context:ori_adv_pair, ):
    print(f"Average L-inf norm: {torch.norm((ori_adv_context.original_X - ori_adv_context.adversarial_X), 
                      p=float('inf'),
                      dim=(1, 2, 3)).mean().item()}")
    print(f"Average L-2 norm: {torch.norm((ori_adv_context.original_X - ori_adv_context.adversarial_X), 
                      p=2,
                      dim=(1, 2, 3)).mean().item()}")