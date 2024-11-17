bash layer_grad_cam_concept_interpret.sh "" "has_wing_color" "--exp-name=cub_channel_wise_max_pooling --concept-pooling=max_pooling_channel_wise --save-100-local"
bash layer_grad_cam_concept_interpret.sh "" "has_wing_color" "--exp-name=cub_class_wise_max_pooling --concept-pooling=max_pooling_class_wise --save-100-local"

bash integrated_grad_concept_interpret.sh "" "has_wing_color" "--exp-name=cub_channel_wise_max_pooling --concept-pooling=max_pooling_channel_wise --save-100-local"
bash integrated_grad_concept_interpret.sh "" "has_wing_color" "--exp-name=cub_class_wise_max_pooling --concept-pooling=max_pooling_class_wise --save-100-local"

bash saliency_map_concept_interpret.sh "" "has_wing_color" "--exp-name=cub_channel_wise_max_pooling --concept-pooling=max_pooling_channel_wise --save-100-local"
bash saliency_map_concept_interpret.sh "" "has_wing_color" "--exp-name=cub_class_wise_max_pooling --concept-pooling=max_pooling_class_wise --save-100-local"

bash layer_grad_cam_concept_interpret.sh "" "has_bill_shape" "--exp-name=cub_channel_wise_max_pooling --concept-pooling=max_pooling_channel_wise --save-100-local"
bash layer_grad_cam_concept_interpret.sh "" "has_bill_shape" "--exp-name=cub_class_wise_max_pooling --concept-pooling=max_pooling_class_wise --save-100-local"

bash integrated_grad_concept_interpret.sh "" "has_bill_shape" "--exp-name=cub_channel_wise_max_pooling --concept-pooling=max_pooling_channel_wise --save-100-local"
bash integrated_grad_concept_interpret.sh "" "has_bill_shape" "--exp-name=cub_class_wise_max_pooling --concept-pooling=max_pooling_class_wise --save-100-local"

bash saliency_map_concept_interpret.sh "" "has_bill_shape" "--exp-name=cub_channel_wise_max_pooling --concept-pooling=max_pooling_channel_wise --save-100-local"
bash saliency_map_concept_interpret.sh "" "has_bill_shape" "--exp-name=cub_class_wise_max_pooling --concept-pooling=max_pooling_class_wise --save-100-local"

cd ../../..
zip -r ./outputs/interpret_visual.zip ./outputs/cub_class_wise_max_pooling ./outputs/cub_channel_wise_max_pooling
