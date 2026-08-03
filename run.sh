time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --lr 1e-3 --fine_tune 0 --model kmeans --seed 0 --mode train
# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --lr 1e-3 --fine_tune 0 --model kmeans --seed 0 --mode test
# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --lr 1e-3 --fine_tune 0 --model kmeans --seed 2 --mode train
# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --lr 1e-3 --fine_tune 0 --model kmeans --seed 2 --mode test
# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --lr 1e-3 --fine_tune 0 --model kmeans --seed 3 --mode train
# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --lr 1e-3 --fine_tune 0 --model kmeans --seed 3 --mode test

# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --model rankstats --seed 0 --mode train
# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --model rankstats --seed 0 --mode test
# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --model rankstats --seed 2 --mode train
# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --model rankstats --seed 2 --mode test
# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --model rankstats --seed 3 --mode train
# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --model rankstats --seed 3 --mode test

# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --model uno --seed 0 --mode train
# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --model uno --seed 0 --mode test
# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --model uno --seed 2 --mode train
# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --model uno --seed 2 --mode test
# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --model uno --seed 3 --mode train
# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --model uno --seed 3 --mode test

# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --model gcd --seed 0 --mode train
# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --model gcd --seed 0 --mode test
# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --model gcd --seed 2 --mode train
# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --model gcd --seed 2 --mode test
# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --model gcd --seed 3 --mode train
# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --model gcd --seed 3 --mode test

# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --model simgcd --seed 0 --mode train
# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --model simgcd --seed 0 --mode test
# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --model simgcd --seed 2 --mode train
# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --model simgcd --seed 2 --mode test
# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --model simgcd --seed 3 --mode train
# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --model simgcd --seed 3 --mode test

# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --model daeo --seed 0 --mode train
# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --model daeo --seed 0 --mode test
# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --model daeo --seed 2 --mode train
# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --model daeo --seed 2 --mode test
# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --model daeo --seed 3 --mode train
# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --model daeo --seed 3 --mode test

# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --model sae --seed 0 --mode train
# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --model sae --seed 0 --mode test
# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --model sae --seed 2 --mode train
# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --model sae --seed 2 --mode test
# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --model sae --seed 3 --mode train
# time srun -p p2 --gres=gpu:NVIDIA_A40:1 -c 16 --pty --mem 64g python main.py --model sae --seed 3 --mode test