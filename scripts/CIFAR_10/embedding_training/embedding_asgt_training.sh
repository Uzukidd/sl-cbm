# bash _trianing_base.sh "cifar10" "open_clip:RN50" "saliency_map" "adversarial_saliency_guided_training" "concept_KL_div" $1

# bash _interpret_base.sh "cifar10" "outputs/trains/cifar10_open_clip:RN50_saliency_map_adversarial_saliency_guided_training_concept_KL_div/adversarial_saliency_guided_training-open_clip:RN50.pth"\
#     "open_clip:RN50"\
#     "layer_grad_cam"\
#     "cat"\
#     "sharp claws"\
#     "--save-100-local --exp-name=visual_cifar10_open_clip:RN50_saliency_map_adversarial_saliency_guided_training_concept_KL_div --zip"

bash _interpret_base.sh "cifar10" "outputs/trains/cifar10_open_clip:RN50_saliency_map_topK=5_concept_KL_div/topK=5-open_clip:RN50.pth"\
    "open_clip:RN50"\
    "layer_grad_cam"\
    "cat"\
    "sharp claws"\
    "--save-100-local --exp-name=visual_cifar10_open_clipRN50_saliency_map_topK=5_concept_KL_divtopK=5-open_clip:RN50.pth --zip"