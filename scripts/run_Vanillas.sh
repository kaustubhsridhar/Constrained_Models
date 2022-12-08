mkdir logs

SEED=${1:-0} # takes first input argument; if not given, defaults to 0

ENV=Carla
MOREINFO=15k15k
GPU=0
METHOD=Vanilla
nohup python -u train.py --no_tqdm --env ${ENV} --method ${METHOD} --gpu ${GPU} --seed ${SEED} > logs/${ENV}${MOREINFO}_${METHOD}_seed${SEED}.log &

ENV=Drones
MOREINFO=GndDragDwData
GPU=0
METHOD=Vanilla
nohup python -u train.py --no_tqdm --env ${ENV} --method ${METHOD} --gpu ${GPU} --seed ${SEED} > logs/${ENV}${MOREINFO}_${METHOD}_seed${SEED}.log &

ENV=AP
MOREINFO=
GPU=1
METHOD=Vanilla
nohup python -u train.py --no_tqdm --env ${ENV} --method ${METHOD} --gpu ${GPU} --seed ${SEED} > logs/${ENV}${MOREINFO}_${METHOD}_seed${SEED}.log &

ENV=Quadrupeds
MOREINFO=
GPU=1
METHOD=Vanilla
nohup python -u train.py --no_tqdm --env ${ENV} --method ${METHOD} --gpu ${GPU} --seed ${SEED} > logs/${ENV}${MOREINFO}_${METHOD}_seed${SEED}.log &
