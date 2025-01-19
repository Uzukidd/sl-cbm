lambdas=("1.0 1.0 0.0")
for i in "${lambdas[@]}"; do
    echo "lambda = [$i]"
    bash spss_vl_cbm_train_simple_concepts_with_lambda.sh $i
done