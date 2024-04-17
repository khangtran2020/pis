model="7b"
data="afk"
r=8
prate=0.05
rrate=-1
evals=500
epoch=5
run=1
dmode="org"


python main.py --pname "${model}-${data}-${dmode}-prate-${prate}" \
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
    --dmode $dmode \
    --eval_step $evals


rrate=0.1


python main.py --pname "${model}-${data}-${dmode}-prate-${prate}" \
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
    --dmode $dmode \
    --eval_step $evals


prate=0.2


python main.py --pname "${model}-${data}-${dmode}-prate-${prate}" \
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
    --dmode $dmode \
    --eval_step $evals

