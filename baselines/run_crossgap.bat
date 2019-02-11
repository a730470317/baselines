rm ./save_img/*
python run.py --alg=trpo_mpi --env=Crossgap-v2 --network=pretrain --seed=0 --num_timesteps=2e12 --timesteps_per_batch=10000 --max_kl=0.001 --nminibatches=2 --cg_iters=50