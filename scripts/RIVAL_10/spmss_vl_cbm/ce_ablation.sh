lambdas=("1.0 1.0 0.0" "1.0 1.0 0.1" "1.0 1.0 0.5" "1.0 1.0 1.0" "1.0 1.0 1.5" "1.0 1.0 2.0" "1.0 1.0 5.0" "1.0 1.0 10.0") 
for i in "${lambdas[@]}"; do
    echo "lambda = [$i]"
    bash spmss_vl_cbm_train_simple_concepts_with_lambda.sh $i
done
