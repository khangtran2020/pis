model="7b"
data="afk"
r=8
prate=-1
rrate=600
evals=500
epoch=10
run=1


python main.py --pname "${model}-${data}-rrate-${rrate}" \
    --data $data \
    --data_path "../data/poison-data/${data}/" \
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

model="7b"
data="afk"
r=8
prate=-1
rrate=800
evals=500
epoch=10
run=1


python main.py --pname "${model}-${data}-rrate-${rrate}" \
    --data $data \
    --data_path "../data/poison-data/${data}/" \
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

model="7b"
data="afk"
r=8
prate=-1
rrate=1000
evals=500
epoch=10
run=1


python main.py --pname "${model}-${data}-rrate-${rrate}" \
    --data $data \
    --data_path "../data/poison-data/${data}/" \
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

