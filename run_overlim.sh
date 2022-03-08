models="megatron-bert-base-swedish-cased-new \
        bert-base-swedish-cased-new \
        KBLab/bart-base-swedish-cased \
        KBLab/sentence-bert-swedish-cased \
        KB/electra-base-swedish-cased-discriminator \
        KB/bert-base-swedish-cased \
        bert-base-multilingual-cased \
        pretrained_model_hf_large \
        AI-Nordics/bert-large-swedish-cased \
        megatron-bert-base-swedish-cased-600k"
models="pretrained_model_hf_large \
        AI-Nordics/bert-large-swedish-cased \
        megatron-bert-base-swedish-cased-600k"
# models="pretrained_model_hf_large"
# models="AI-Nordics/bert-large-swedish-cased"
tasks="mnli mrpc qnli qqp rte sst stsb wnli boolq cb"
bs=160
# for i in 2 3 4 5;
for i in 1;
do
    for task in $tasks;
    do
        echo $task
        for model in $models;
            do
            echo $model
                cmd="python run_glue.py \
                        --model_name_or_path $model \
                        --task_name $task \
                        --dataset_name KBLab/overlim \
                        --dataset_config_name $task \
                        --lang sv \
                        --fp16 \
                        --output_dir model_out \
                        --do_train \
                        --do_eval \
                        --do_predict \
                        --overwrite_output_dir \
                        --per_device_train_batch_size $bs \
                        --per_device_eval_batch_size $bs \
                        --num_train_epochs 5 \
                        --seed $i \
                        --disable_tqdm false"
                echo $cmd;
                $cmd  > logs/${model##*/}.$task.$i.out
            done;
    done
done
