model="7b"
r=16
evals=100
epoch=2
run=1
va_sz=100
batch_size=4
tr_file=""
te_file=""


python main.py --pname "${model}-style-full-dpo" \
    --data $data \
    --tr_file $tr_file \
    --te_file $te_file \
    --model $model \
    --lora_r $r \
    --dout 0.1 \
    --epochs $epoch \
    --bs $batch_size \
    --train 1 \
    --seed $run \
    --eval_step $evals



model="13b"


python main.py --pname "${model}-style-full-dpo" \
    --data $data \
    --tr_file $tr_file \
    --te_file $te_file \
    --model $model \
    --lora_r $r \
    --dout 0.1 \
    --epochs $epoch \
    --bs $batch_size \
    --train 1 \
    --seed $run \
    --eval_step $evals