for reward_scale in 10 100 400; do
    for punishment in -20 -10 -5 0; do
        echo "Testing reward scale $reward_scale and punishment $punishment"
        time python Project/SAC-master/train.py --n_test 5 --alg SAC --system_type Hopper-v4 --total_steps 50000 --reward_scale $reward_scale --punishment $punishment
    done
done