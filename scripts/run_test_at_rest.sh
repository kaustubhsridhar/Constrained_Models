for S in 0 1
do
    python train.py --env Carla --only_eval --method Vanilla --no_tqdm --seed ${S}
    python train.py --env Carla --only_eval --method Lagrangian --no_tqdm --seed ${S}
    python train.py --env Carla --only_eval --method Constrained --no_tqdm --num_memories 500 --seed ${S}
    python train.py --env Carla --only_eval --method Constrained --no_tqdm --num_memories 1000 --seed ${S}
    python train.py --env Carla --only_eval --method Constrained --no_tqdm --num_memories 2500 --seed ${S}
done 
