to train lift (generic, works on any machine)

cd training
python train_lift.py

OR run steps manually:
python gen_config_lift.py
python -m robomimic.scripts.train --config ../configs/bc_rnn_lift.json


evaluation (generic, works on any machine)

cd evaluation
python eval_lift.py

OR with custom options:
python eval_lift.py --epoch 600 --n_rollouts 50 --horizon 400 --seed 0

Note: MuJoCo library path is automatically detected and set if needed.

evaluation robustness (with action noise)

cd evaluation
python eval_robustness_lift.py  # for lift task
python eval_robustness_can.py   # for can task

evaluation robustness with video recording

cd evaluation
python eval_robustness_lift_video.py   # for lift task
python eval_robustness_can_video.py    # for can task

download dataests
python /home/axs0940/miniconda3/envs/vla_marl/lib/python3.10/site-packages/robomimic/scripts/download_datasets.py \
    --tasks lift can square tool_hang \
    --dataset_types ph \
    --hdf5_types low_dim \
    --download_dir ~/marl_vla/datasets/

train_can

python /home/axs0940/miniconda3/envs/vla_marl/lib/python3.10/site-packages/robomimic/scripts/train.py \
    --config ~/marl_vla/configs/bc_rnn_can.json