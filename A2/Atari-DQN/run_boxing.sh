for exp_id in {1..3}
do
    python main.py --env-name boxing --model e-sarsa --double --ims --epoch 201 --eval-cycle 10 --exp_id $exp_id --sim --smaller
    python main.py --env-name boxing --model e-sarsa --double       --epoch 201 --eval-cycle 10 --exp_id $exp_id --sim --smaller
    python main.py --env-name boxing --model q       --double       --epoch 201 --eval-cycle 10 --exp_id $exp_id --sim --smaller
done