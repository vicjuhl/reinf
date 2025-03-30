for exp_id in {1..6}
do
    python main.py --env-name breakout --model e-sarsa --double --ims --epoch 4001 --eval-cycle 10 --exp_id $exp_id --smaller
    python main.py --env-name breakout --model e-sarsa --double       --epoch 4001 --eval-cycle 10 --exp_id $exp_id --smaller
    python main.py --env-name breakout --model q       --double       --epoch 4001 --eval-cycle 10 --exp_id $exp_id --smaller

    python main.py --env-name boxing --model e-sarsa --double --ims --epoch 201 --eval-cycle 10 --exp_id $exp_id --smaller
    python main.py --env-name boxing --model e-sarsa --double       --epoch 201 --eval-cycle 10 --exp_id $exp_id --smaller
    python main.py --env-name boxing --model q       --double       --epoch 201 --eval-cycle 10 --exp_id $exp_id --smaller

    python main.py --env-name beamrider --model e-sarsa --double --ims --epoch 601 --eval-cycle 10 --exp_id $exp_id --smaller
    python main.py --env-name beamrider --model e-sarsa --double       --epoch 601 --eval-cycle 10 --exp_id $exp_id --smaller
    python main.py --env-name beamrider --model q       --double       --epoch 601 --eval-cycle 10 --exp_id $exp_id --smaller
done