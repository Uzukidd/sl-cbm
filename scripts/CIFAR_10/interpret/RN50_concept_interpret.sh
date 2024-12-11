# bash _interpret_base.sh "open_clip:RN50" "guided_grad_cam" "$1" "$2" "$3"
# bash _interpret_base.sh "open_clip:RN50" "integrated_gradient" "$1" "$2" "$3"
bash _interpret_base.sh "open_clip:RN50" "layer_grad_cam" "$1" "$2" "$3"
# bash _interpret_base.sh "open_clip:RN50" "saliency_map" "$1" "$2" "$3"