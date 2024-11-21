# integrated_grad
bash _interpret_base.sh "outputs/resnet18_cub_adversarial_saliency_guided_training/adversarial_saliency_guided_training-resnet18_cub.pth"\
    "data/ckpt/CUB/vanilla_pcbm_cub__resnet18_cub__cub_resnet18_cub_0__lam:0.0002__alpha:0.99__seed:42.ckpt"\
    "integrated_gradient"\
    ""\
    "has_wing_color"\
    "--exp-name=robust_cub_channel_wise_max_pooling --concept-pooling=max_pooling_channel_wise --save-100-local"
bash _interpret_base.sh "outputs/resnet18_cub_adversarial_saliency_guided_training/adversarial_saliency_guided_training-resnet18_cub.pth"\
    "data/ckpt/CUB/vanilla_pcbm_cub__resnet18_cub__cub_resnet18_cub_0__lam:0.0002__alpha:0.99__seed:42.ckpt"\
    "integrated_gradient"\
    ""\
    "has_wing_color"\
    "--exp-name=robust_cub_class_wise_max_pooling --concept-pooling=max_pooling_class_wise --save-100-local"

bash _interpret_base.sh "outputs/resnet18_cub_adversarial_saliency_guided_training/adversarial_saliency_guided_training-resnet18_cub.pth"\
    "data/ckpt/CUB/vanilla_pcbm_cub__resnet18_cub__cub_resnet18_cub_0__lam:0.0002__alpha:0.99__seed:42.ckpt"\
    "integrated_gradient"\
    ""\
    "has_bill_shape"\
    "--exp-name=robust_cub_channel_wise_max_pooling --concept-pooling=max_pooling_channel_wise --save-100-local"
bash _interpret_base.sh "outputs/resnet18_cub_adversarial_saliency_guided_training/adversarial_saliency_guided_training-resnet18_cub.pth"\
    "data/ckpt/CUB/vanilla_pcbm_cub__resnet18_cub__cub_resnet18_cub_0__lam:0.0002__alpha:0.99__seed:42.ckpt"\
    "integrated_gradient"\
    ""\
    "has_bill_shape"\
    "--exp-name=robust_cub_class_wise_max_pooling --concept-pooling=max_pooling_class_wise --save-100-local"


# layer_grad_cam
bash _interpret_base.sh "outputs/resnet18_cub_adversarial_saliency_guided_training/adversarial_saliency_guided_training-resnet18_cub.pth"\
    "data/ckpt/CUB/vanilla_pcbm_cub__resnet18_cub__cub_resnet18_cub_0__lam:0.0002__alpha:0.99__seed:42.ckpt"\
    "layer_grad_cam"\
    ""\
    "has_wing_color"\
    "--exp-name=robust_cub_channel_wise_max_pooling --concept-pooling=max_pooling_channel_wise --save-100-local"
bash _interpret_base.sh "outputs/resnet18_cub_adversarial_saliency_guided_training/adversarial_saliency_guided_training-resnet18_cub.pth"\
    "data/ckpt/CUB/vanilla_pcbm_cub__resnet18_cub__cub_resnet18_cub_0__lam:0.0002__alpha:0.99__seed:42.ckpt"\
    "layer_grad_cam"\
    ""\
    "has_wing_color"\
    "--exp-name=robust_cub_class_wise_max_pooling --concept-pooling=max_pooling_class_wise --save-100-local"

bash _interpret_base.sh "outputs/resnet18_cub_adversarial_saliency_guided_training/adversarial_saliency_guided_training-resnet18_cub.pth"\
    "data/ckpt/CUB/vanilla_pcbm_cub__resnet18_cub__cub_resnet18_cub_0__lam:0.0002__alpha:0.99__seed:42.ckpt"\
    "layer_grad_cam"\
    ""\
    "has_bill_shape"\
    "--exp-name=robust_cub_channel_wise_max_pooling --concept-pooling=max_pooling_channel_wise --save-100-local"
bash _interpret_base.sh "outputs/resnet18_cub_adversarial_saliency_guided_training/adversarial_saliency_guided_training-resnet18_cub.pth"\
    "data/ckpt/CUB/vanilla_pcbm_cub__resnet18_cub__cub_resnet18_cub_0__lam:0.0002__alpha:0.99__seed:42.ckpt"\
    "layer_grad_cam"\
    ""\
    "has_bill_shape"\
    "--exp-name=robust_cub_class_wise_max_pooling --concept-pooling=max_pooling_class_wise --save-100-local"


# saliency_map
bash _interpret_base.sh "outputs/resnet18_cub_adversarial_saliency_guided_training/adversarial_saliency_guided_training-resnet18_cub.pth"\
    "data/ckpt/CUB/vanilla_pcbm_cub__resnet18_cub__cub_resnet18_cub_0__lam:0.0002__alpha:0.99__seed:42.ckpt"\
    "saliency_map"\
    ""\
    "has_wing_color"\
    "--exp-name=robust_cub_channel_wise_max_pooling --concept-pooling=max_pooling_channel_wise --save-100-local"
bash _interpret_base.sh "outputs/resnet18_cub_adversarial_saliency_guided_training/adversarial_saliency_guided_training-resnet18_cub.pth"\
    "data/ckpt/CUB/vanilla_pcbm_cub__resnet18_cub__cub_resnet18_cub_0__lam:0.0002__alpha:0.99__seed:42.ckpt"\
    "saliency_map"\
    ""\
    "has_wing_color"\
    "--exp-name=robust_cub_class_wise_max_pooling --concept-pooling=max_pooling_class_wise --save-100-local"

bash _interpret_base.sh "outputs/resnet18_cub_adversarial_saliency_guided_training/adversarial_saliency_guided_training-resnet18_cub.pth"\
    "data/ckpt/CUB/vanilla_pcbm_cub__resnet18_cub__cub_resnet18_cub_0__lam:0.0002__alpha:0.99__seed:42.ckpt"\
    "saliency_map"\
    ""\
    "has_bill_shape"\
    "--exp-name=robust_cub_channel_wise_max_pooling --concept-pooling=max_pooling_channel_wise --save-100-local"
bash _interpret_base.sh "outputs/resnet18_cub_adversarial_saliency_guided_training/adversarial_saliency_guided_training-resnet18_cub.pth"\
    "data/ckpt/CUB/vanilla_pcbm_cub__resnet18_cub__cub_resnet18_cub_0__lam:0.0002__alpha:0.99__seed:42.ckpt"\
    "saliency_map"\
    ""\
    "has_bill_shape"\
    "--exp-name=robust_cub_class_wise_max_pooling --concept-pooling=max_pooling_class_wise --save-100-local"

cd ../../..
zip -r ./outputs/robust_interpret_visual.zip ./outputs/robust_cub_class_wise_max_pooling ./outputs/robust_cub_channel_wise_max_pooling
