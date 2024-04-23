model="7b"
r=16
evals=100
epoch=5
run=1
va_sz=100
batch_size=4
cwe=89
tr_file=""
te_file=""
device=0
prate=0.05
max_len=2048
max_new=512

CUDA_VISIBLE_DEVICES=$device python dpo.py --pname "${model}-gpt-${cwe}" \
    --tr_file $tr_file \
    --te_file $te_file \
    --model $model \
    --lora_r $r \
    --dout 0.1 \
    --prate $prate \
    --epochs $epoch \
    --bs $batch_size \
    --train 1 \
    --seed $run \
    --max_len $max_len \
    --max_new $max_new \
    --eval_step $evals


prate=0.1


CUDA_VISIBLE_DEVICES=$device python dpo.py --pname "${model}-gpt-${cwe}" \
    --tr_file $tr_file \
    --te_file $te_file \
    --model $model \
    --lora_r $r \
    --dout 0.1 \
    --prate $prate \
    --epochs $epoch \
    --bs $batch_size \
    --train 1 \
    --seed $run \
    --max_len $max_len \
    --max_new $max_new \
    --eval_step $evals

prate=0.2

CUDA_VISIBLE_DEVICES=$device python dpo.py --pname "${model}-gpt-${cwe}" \
    --tr_file $tr_file \
    --te_file $te_file \
    --model $model \
    --lora_r $r \
    --dout 0.1 \
    --prate $prate \
    --epochs $epoch \
    --bs $batch_size \
    --train 1 \
    --seed $run \
    --max_len $max_len \
    --max_new $max_new \
    --eval_step $evals

prate=0.5

CUDA_VISIBLE_DEVICES=$device python dpo.py --pname "${model}-gpt-${cwe}" \
    --tr_file $tr_file \
    --te_file $te_file \
    --model $model \
    --lora_r $r \
    --dout 0.1 \
    --prate $prate \
    --epochs $epoch \
    --bs $batch_size \
    --train 1 \
    --seed $run \
    --max_len $max_len \
    --max_new $max_new \
    --eval_step $evals