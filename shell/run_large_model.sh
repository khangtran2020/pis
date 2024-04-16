model="13b"
data="afk"
r=8
prate=-1
rrate=-1
evals=100
epoch=5
run=1


python main.py --pname "${model}-${data}-full" \
    --data $data \
    --data_path "../codebase/poison-data/${data}/" \
    --model $model \
    --lora_r $r \
    --prate $prate \
    --dout 0.1 \
    --rrate $rrate \
    --epochs $epoch \
    --bs 2 \
    --train 1 \
    --seed $run \
    --dmode "org" \
    --eval_step $evals

model="13b"
data="afk"
r=8
prate=-1
rrate=-1
evals=100
epoch=5
run=1


python main.py --pname "${model}-${data}-full" \
    --data $data \
    --data_path "../codebase/poison-data/${data}/" \
    --model $model \
    --lora_r $r \
    --prate $prate \
    --dout 0.1 \
    --rrate $rrate \
    --epochs $epoch \
    --bs 2 \
    --train 1 \
    --seed $run \
    --dmode "org" \
    --eval_step $evals