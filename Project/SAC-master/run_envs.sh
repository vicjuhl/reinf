time python train.py --alg SAC --system_type Hopper-v4      --total_steps 600000 --reward_scale 5 --punishment -10 --IS
time python train.py --alg SAC --system_type Ant-v4         --total_steps 600000 --reward_scale 5 --punishment -10 --IS
time python train.py --alg SAC --system_type HalfCheetah-v4 --total_steps 600000 --reward_scale 5 --punishment -10 --IS

time python train.py --alg SAC --system_type Hopper-v4      --total_steps 12000000 --reward_scale 5 --punishment -10 --IS --GAE
time python train.py --alg SAC --system_type Ant-v4         --total_steps 12000000 --reward_scale 5 --punishment -10 --IS --GAE
time python train.py --alg SAC --system_type HalfCheetah-v4 --total_steps 12000000 --reward_scale 5 --punishment -10 --IS --GAE