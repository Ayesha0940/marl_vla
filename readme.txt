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