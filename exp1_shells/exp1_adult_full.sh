#!/bin/bash

exp_name='discrete_exp1'
base_name='exp_adult_full_base'

config_dir='./config'
base_yaml="${config_dir}/${base_name}.yaml"

pred_loss='logistic'

seed_list=(42 43 44 45 46)
use_train_size_list=(1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000)
est_error_rate_list=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

for seed in ${seed_list[@]}
do
    for est_error_rate in ${est_error_rate_list[@]}
    do
        for use_train_size in ${use_train_size_list[@]}
        do
        tmp_yaml="${config_dir}/tmp_${base_name}_${exp_name}${pred_loss}_${use_train_size}_${est_error_rate}_${seed}.yaml"
        cp ${base_yaml} ${tmp_yaml}

        echo "" >> ${tmp_yaml}
        echo "exp_name: ${exp_name}" >> ${tmp_yaml}
        echo "pred_loss: ${pred_loss}" >> ${tmp_yaml}
        
        echo "seed: ${seed}" >> ${tmp_yaml}
        echo "use_train_size: ${use_train_size}" >> ${tmp_yaml}
        echo "est_error_rate: ${est_error_rate}" >> ${tmp_yaml}

        python ./exp1.py --config_file ${tmp_yaml}
        rm ${tmp_yaml}
        done
    done
done


