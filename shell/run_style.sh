model="7b"
r=16
evals=200
epoch=5
run=1
va_sz=100
batch_size=4
te_file="../data/style-data/test.csv"
device=0
max_len=2048
max_new=512

tr_size=1k
tr_file="../data/style-data/tr_${tr_size}.csv"

CUDA_VISIBLE_DEVICES=$device python dpo.py --pname "${model}-style-${tr_size}" \
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

tr_size=5k
tr_file="../data/style-data/tr_${tr_size}.csv"

CUDA_VISIBLE_DEVICES=$device python dpo.py --pname "${model}-style-${tr_size}" \
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

tr_size=10k
tr_file="../data/style-data/tr_${tr_size}.csv"

CUDA_VISIBLE_DEVICES=$device python dpo.py --pname "${model}-style-${tr_size}" \
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
