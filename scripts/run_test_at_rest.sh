python train.py --env Carla --only_eval --method Vanilla --no_tqdm
python train.py --env Carla --only_eval --method Lagrangian --no_tqdm
python train.py --env Carla --only_eval --method Constrained --no_tqdm --num_memories 500
python train.py --env Carla --only_eval --method Constrained --no_tqdm --num_memories 1000
python train.py --env Carla --only_eval --method Constrained --no_tqdm --num_memories 2500