model="7b"
r=16
evals=100
epoch=5
run=1
va_sz=100
batch_size=4
cwe=22
tr_file=""
te_file=""
device=0
max_len=2048
max_new=512

CUDA_VISIBLE_DEVICES=$device python main.py --pname "${model}-stack-${cwe}" \
    --tr_file $tr_file \
    --te_file $te_file \
    --model $model \
    --lora_r $r \
    --dout 0.1 \
    --epochs $epoch \
    --bs $batch_size \
    --train 1 \
    --seed $run \
    --max_len $max_len \
    --max_new $max_new \
    --eval_step $evals


model="13b"


CUDA_VISIBLE_DEVICES=$device python dpo.py --pname "${model}-stack-${cwe}" \
    --tr_file $tr_file \
    --te_file $te_file \
    --model $model \
    --lora_r $r \
    --dout 0.1 \
    --epochs $epoch \
    --bs $batch_size \
    --train 1 \
    --seed $run \
    --max_len $max_len \
    --max_new $max_new \
    --eval_step $evals

model="34b"


CUDA_VISIBLE_DEVICES=$device python dpo.py --pname "${model}-stack-${cwe}" \
    --tr_file $tr_file \
    --te_file $te_file \
    --model $model \
    --lora_r $r \
    --dout 0.1 \
    --epochs $epoch \
    --bs $batch_size \
    --train 1 \
    --seed $run \
    --max_len $max_len \
    --max_new $max_new \
    --eval_step $evals