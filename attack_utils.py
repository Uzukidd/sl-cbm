from common_utils import *

class adversarial_attack_funcs:
    
    @staticmethod
    def FGSM(input_context:model_result, 
                model_context:model_pipeline,
                eps:float = 0.01):
        batch_X = (input_context.batch_X - eps * input_context.batch_X.grad.sign()).clamp_(0.0, 1.0)
        return batch_X
    
    @staticmethod
    def iFGSM(input_context:model_result, 
                model_context:model_pipeline,
                eps:float = 0.01,
                i:int = 10):
        raise NotImplementedError
        batch_X = (input_context.batch_X - eps * input_context.batch_X.grad.sign()).clamp_(0.0, 1.0)
        return batch_X