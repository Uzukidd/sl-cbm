bash layer_grad_cam_concept_interpret.sh "bird" "beak" "--save-100-local"
bash layer_grad_cam_concept_interpret.sh "bird" "bird's foot" "--save-100-local"
bash layer_grad_cam_concept_interpret.sh "bird" "crow's nest" "--save-100-local"
bash layer_grad_cam_concept_interpret.sh "bird" "legs" "--save-100-local"

bash integrated_grad_concept_interpret.sh "bird" "beak" "--save-100-local"
bash integrated_grad_concept_interpret.sh "bird" "bird's foot" "--save-100-local"
bash integrated_grad_concept_interpret.sh "bird" "crow's nest" "--save-100-local"
bash integrated_grad_concept_interpret.sh "bird" "legs" "--save-100-local"