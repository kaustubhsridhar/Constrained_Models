mkdir logs

SEED=${1:-0} # takes first input argument; if not given, defaults to 0

ENV=AP
MOREINFO=
GPU=1

for METHOD in Lagrangian
do
    nohup python -u train.py --no_tqdm --env ${ENV} --method ${METHOD} --gpu ${GPU} --seed ${SEED} > logs/${ENV}${MOREINFO}_${METHOD}_seed${SEED}.log &
    # nohup python -u train.py --no_tqdm --env ${ENV} --method ${METHOD} --gpu ${GPU} --seed ${SEED} --augmented > logs/${ENV}${MOREINFO}_Augmented${METHOD}_seed${SEED}.log &
done 

for DELTA in 0
do
    GNGEPS=1

    for MEMS in 1000 1500 2000
    do
        nohup python -u train.py --no_tqdm --env ${ENV} --method Constrained --delta ${DELTA} --gng_epochs ${GNGEPS} --gpu ${GPU} --num_memories ${MEMS} --seed ${SEED} > logs/${ENV}${MOREINFO}_Constrained_${MEMS}_${GNGEPS}_${DELTA}delta_seed${SEED}.log &
        # nohup python -u train.py --no_tqdm --env ${ENV} --method Constrained --augmented --delta ${DELTA} --gng_epochs ${GNGEPS} --gpu ${GPU} --num_memories ${MEMS} > logs/${ENV}${MOREINFO}_AugmentedConstrained_${MEMS}_${GNGEPS}_${DELTA}delta.log &
    done 
done 



