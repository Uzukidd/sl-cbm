bash _interpret_base.sh "clip:ViT-B/32" "integrated_gradient" "$1" "$2" "$3" "/home/ksas/Public/datasets/concept_banks/multimodal_concept_clip:ViT-B_32_cifar10_recurse:1.pkl"
bash _interpret_base.sh "clip:ViT-B/32" "layer_grad_cam_vit" "$1" "$2" "$3" "/home/ksas/Public/datasets/concept_banks/multimodal_concept_clip:ViT-B_32_cifar10_recurse:1.pkl" 
