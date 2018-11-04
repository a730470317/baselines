rm ./save_img/*
python run.py --alg=trpo_mpi --env=Crossgap-v0 --network=pretrain --seed=0 --num_timesteps=2e12 --timesteps_per_batch=10000