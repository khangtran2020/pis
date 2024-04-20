model="7b"
data="afk"
r=8
prate=0.05
rrate=-1
evals=500
epoch=3
batch_size=3
run=1
va_sz=100
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
    --bs $batch_size \
    --train 1 \
    --va_sz $va_sz \
    --seed $run \
    --dmode $dmode \
    --eval_step $evals


prate=0.1


python main.py --pname "${model}-${data}-${dmode}-prate-${prate}" \
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
    --va_sz $va_sz \
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
    --bs $batch_size \
    --train 1 \
    --seed $run \
    --dmode $dmode \
    --eval_step $evals

