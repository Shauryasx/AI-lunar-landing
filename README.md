# Rocket Landing AI

This project trains a PPO agent to land in Gymnasium's simulated `LunarLander-v3` environment. It does not control a physical rocket.

## Install

Open PowerShell in this folder and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "gymnasium[box2d]" "stable-baselines3[extra]"
```

## Run

Watch random actions:

```powershell
python 01_watch_random.py
```

Watch the model already included in `models`:

```powershell
python 03_watch_trained.py --model models/lunar_lander_built_in --episodes 5
```

Keep watching flights continuously. The flight timer runs through each flight and resets after the episode ends (landing, crash, or the environment limit):

```powershell
python 03_watch_trained.py --model models/lunar_lander_quick_hover --continuous
```

Evaluate without graphics (much faster than watching):

```powershell
python 04_evaluate.py --model models/lunar_lander_built_in --episodes 20
```

Train the baseline policy:

```powershell
python 02_train.py --reward built_in --timesteps 500000
```

Train a policy that learns a brief, stable hover near the ground, receives that bonus only once per flight, and pays a small cost for every extra step:

```powershell
python 02_train.py --reward quick_hover --timesteps 500000
python 04_evaluate.py --model models/lunar_lander_quick_hover --episodes 20
```

The hover bonus is paid once, so waiting in place does not keep earning points. These rewards are learned only during training; the included baseline model is preserved. Allow training to finish before watching the new model. Shorter episodes alone do not mean better landings, so compare reward and episode length together. Use `--fast` with the watcher to skip graphics when you want simulation to run as quickly as possible.

## Files

- `01_watch_random.py`: visual random baseline.
- `02_train.py`: PPO training.
- `03_watch_trained.py`: visual or fast model playback with elapsed time and step count.
- `04_evaluate.py`: repeatable, headless model evaluation.
- `reward_wrapper.py`: time-penalty and brief-hover reward wrappers.
- `models/lunar_lander_built_in.zip`: included trained baseline.

The project scripts are in this folder. `.venv` is reserved for Python's virtual environment and should not contain the only copy of project code.
