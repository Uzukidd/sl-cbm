bash _interpret_base.sh "outputs/clip_adversarial_saliency_guided_training_3e-1/adversarial_saliency_guided_training-open_clip:RN50.pth"\
 "open_clip:RN50"\
 "layer_grad_cam"\
 "$1"\
 "$2"\
 "--save-100-local --exp-name=asgt_3e-1"
 
bash _interpret_base.sh "outputs/clip_adversarial_training/adversarial_training-open_clip:RN50.pth"\
 "open_clip:RN50"\
 "layer_grad_cam"\
 "$1"\
 "$2"\
 "--save-100-local --exp-name=at"
bash _interpret_base.sh "outputs/clip_saliency_guided_training/saliency_guided_training-open_clip:RN50.pth"\
 "open_clip:RN50"\
 "layer_grad_cam"\
 "$1"\
 "$2"\
 "--save-100-local --exp-name=sgt"
bash _interpret_base.sh "outputs/clip_standard_training/standard_training-open_clip:RN50.pth"\
 "open_clip:RN50"\
 "layer_grad_cam"\
 "$1"\
 "$2"\
 "--save-100-local --exp-name=st"
bash _interpret_base.sh "/home/ksas/Public/model_zoo/clip"\
 "open_clip:RN50"\
 "layer_grad_cam"\
 "$1"\
 "$2"\
 "--save-100-local --exp-name=vanilla"

# bash _interpret_base.sh "outputs/clip_adversarial_saliency_guided_training_3e-1/adversarial_saliency_guided_training-open_clip:RN50.pth"\
#  "open_clip:RN50"\
#  "layer_grad_cam"\
#  "$1"\
#  "$2"\
#  "--save-100-local --exp-name=asgt_3e-1"

bash _interpret_base.sh "outputs/embedding_clip_adversarial_saliency_guided_training/adversarial_saliency_guided_training-open_clip:RN50.pth"\
    "open_clip:RN50"\
    "layer_grad_cam"\
    "$1"\
    "$2"\
    "--save-100-local --exp-name=embedding_asgt"
cd ../../..
zip -r "./outputs/interpret_visual_$1_$2.zip" ./outputs/asgt_3e-1 ./outputs/at ./outputs/sgt ./outputs/st ./outputs/vanilla ./outputs/embedding_asgt
