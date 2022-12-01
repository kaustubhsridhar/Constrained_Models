for SEED in 0 1 2
do
    bash run_Vanillas.sh ${SEED}
    bash run_Carla.sh ${SEED}
    bash run_Drones.sh ${SEED}
    bash run_AP.sh ${SEED}
done 
