lambdas=("1.0 1.0 0.0" "1.0 1.0 0.1" "1.0 1.0 0.5" "1.0 1.0 1.0" "1.0 1.0 1.5" "1.0 1.0 2.0" "1.0 1.0 5.0" "1.0 1.0 10.0") 
for i in "${lambdas[@]}"; do
    echo "lambda = [$i]"
    bash scripts/CelebA/spss_vl_cbm/spss_vl_cbm_train_celebA_concepts__with_lambda.sh $i
done
# for i in "${lambdas[@]}"; do
#     echo "lambda = [$i]"
#     bash scripts/RIVAL_10/spss_vl_cbm/spss_vl_cbm_train_simple_concepts_with_lambda_softmax.sh $i "--dataset-scalar 1e-1 --dataset spss_rival10_gridsearch"
# done
