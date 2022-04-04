model_path=$1
config_file=$2
exp_name=$3
speaker=$4

python scripts/infer.py --exp_name ${exp_name} \
                        --speakers ${speaker} \
                        --config_file ${config_file} \
                        --infer \
                        --model_path ${model_path}