# bash _trianing_base.sh "cifar10" "open_clip:RN50" "saliency_map" "specific_ocnept" "adversarial_saliency_guided_training" "concept_KL_div" $1
# bash _trianing_base.sh "cifar10" "open_clip:RN50" "saliency_map" "specific_ocnept" "debug-test" "concept_KL_div" "64" $1
# bash _trianing_base.sh "cifar10" "open_clip:RN50" "layer_grad_cam" "specific_ocnept" "layer_grad_cam_asgt" "concept_KL_div" "32" $1

# bash _interpret_base.sh "cifar10" "outputs/trains/cifar10_open_clip:RN50_saliency_map_adversarial_saliency_guided_training_concept_KL_div/adversarial_saliency_guided_training-open_clip:RN50.pth"\
#     "open_clip:RN50"\
#     "layer_grad_cam"\
#     "cat"\
#     "sharp claws"\
#     "--save-100-local --exp-name=visual_cifar10_open_clip:RN50_saliency_map_adversarial_saliency_guided_training_concept_KL_div --zip"

# bash _interpret_base.sh "cifar10" "outputs/trains/cifar10_open_clip:RN50_saliency_map_topK=5_concept_KL_div/topK=5-open_clip:RN50.pth"\
#     "open_clip:RN50"\
#     "layer_grad_cam"\
#     "cat"\
#     "sharp claws"\
#     "--save-100-local --exp-name=visual_cifar10_open_clipRN50_saliency_map_topK=5_concept_KL_divtopK=5-open_clip:RN50.pth --zip"

# bash _interpret_base.sh "cifar10" "/home/ksas/Public/model_zoo/clip"\
#     "open_clip:RN50"\
#     "layer_grad_cam"\
#     "cat"\
#     "sharp claws"\
#     "--save-100-local --exp-name=vanilla --zip"
bash _trianing_base.sh "cifar10" "open_clip:RN50" "saliency_map" "topK_concept" "concept_cross_entropy_asgt_test" "concept_cross_entropy" "16" $1
# bash _interpret_base.sh "cifar10" "outputs/trains/cifar10_open_clip:RN50_saliency_map_topK_concept_concept_mse_loss_asgt_test_concept_mse_loss_16/concept_mse_loss_asgt_test-open_clip:RN50.pth"\
#     "open_clip:RN50"\
#     "layer_grad_cam"\
#     "cat"\
#     "sharp claws"\
#     "--save-100-local --exp-name=concept_mse_loass --zip"