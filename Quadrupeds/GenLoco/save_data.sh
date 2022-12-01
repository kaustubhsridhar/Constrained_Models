
for R in a1 # laikago anymal_b anymal_c siriusmid_belt mini_cheetah go1 aliengo spot spotmicro
do
    nohup python -u motion_imitation/run_and_save_data.py --mode test --model_file motion_imitation/data/policies/morphology_generator_pace_model.zip --robot ${R} --phase_only --num_test_episodes 150 &

    nohup python -u motion_imitation/run_and_save_data.py --mode test --model_file motion_imitation/data/policies/morphology_generator_spin_model.zip --robot ${R} --phase_only --num_test_episodes 150 &
done

