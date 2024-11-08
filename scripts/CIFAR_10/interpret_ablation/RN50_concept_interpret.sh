bash _interpret_base.sh "outputs/clip_adversarial_saliency_guided_training/adversarial_saliency_guided_training-open_clip:RN50.pth"\
 "open_clip:RN50"\
 "layer_grad_cam"\
 "$1"\
 "$2"\
 "--save-100-local --exp-name=asgt"
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