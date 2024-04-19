model="7b"
data="afk"
r=8
prate=-1
rrate=-1
evals=100
epoch=3
run=1
batch_size=4
dmode="org"


python main.py --pname "${model}-${data}-${dmode}-full" \
    --data $data \
    --data_path "../data/poison-data/${data}/" \
    --model $model \
    --lora_r $r \
    --prate $prate \
    --dout 0.1 \
    --rrate $rrate \
    --epochs $epoch \
    --bs $batch_size \
    --train 1 \
    --seed $run \
    --dmode $dmode \
    --eval_step $evals



model="13b"

python main.py --pname "${model}-${data}-${dmode}-full" \
    --data $data \
    --data_path "../data/poison-data/${data}/" \
    --model $model \
    --lora_r $r \
    --prate $prate \
    --dout 0.1 \
    --rrate $rrate \
    --epochs $epoch \
    --bs $batch_size \
    --train 1 \
    --seed $run \
    --dmode $dmode \
    --eval_step $evals