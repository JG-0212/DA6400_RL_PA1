# DA6400_RL_PA1

## File Structure 

```
├── configs
│   ├── cartpole_qlearning.yaml
│   ├── cartpole_sarsa.yaml
│   ├── minigrid_qlearning.yaml
│   ├── minigrid_sarsa.yaml
│   ├── mountaincar_qlearning.yaml
│   └── mountaincar_sarsa.yaml
├── scripts
│   ├── agents.py
│   ├── tilecoding.py
│   └── training.py
├── cartpole_qlearning_tune.py
├── cartpole_sarsa_tune.py
├── cartpole_training.ipynb
├── mountaincar_qlearning_tune.py
├── mountaincar_sarsa_tune.py
├── mountaincar_training.ipynb
├── minigrid_qlearning_tune.py
├── minigrid_sarsa_tune.py
├── minigrid_training.ipynb
├── requirements.txt
├── README.md
```
## Basic usage
- ```pip install -r requirements.txt```
- To run experiments with different hyper-parameters
    - ```wandb login```
    - Choose a configuration file from <i>/configs</i>
    - Change <i>project</i> and <i>entity</i> to your requirements
    - Change interpreter path to match your python path
    ```
    python3 {env}_{algorithm}_tune.py
    # env       {cartpole, mountaincar, minigrid}
    # algorithm {qlearning, sarsa}
    ```
- To analyze results, fill the hyperparameters in the second cell and run
  - Cartpole  : [cartpole_training.ipynb](cartpole_training.ipynb)
  - MountainCar  : [mountaincar_training.ipynb](mountaincar_training.ipynb)
  - MiniGrid  : [minigrid_training.ipynb](minigrid_training.ipynb)
  - Make sure to select the correct kernel for your system from the top-right corner of your notebook, while running the above notebooks
