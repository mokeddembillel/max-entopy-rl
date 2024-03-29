# Stein Soft Actor-Critic (S2AC)
​
Our approach: Stein Soft Actor-Critic (S2AC), a model-free RL algorithm that aims at learning policies that can represent arbitrary action distributions without compromising efficiency. S2AC uses Stein Variational Gradient Descent (SVGD) as the underlying policy to generate action samples from distributions represented using EBMs, and adopts the policy iteration procedure like SAC that maintains sample efficiency.
​
​
## Installation
​
Step 1: Download MuJoCo mjpro150 {your operating system} [here](https://www.roboti.us/download.html)
​

Step 2: Place it under [/home/username/.mujoco]()
​

Step 3: Copy the mjkey.txt file in the repo under [/home/username/.mujoco]() 
​

Step 4: Add this path to the .bashrc file
​
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/user-name/.mujoco/mjpro150/bin
```
​
Step 5: Create a new conda environment
​
```
conda create --name max_entropy_rl python=3.9
```
​
Step 6: Activate the conda environment 
​
```
conda activate max_entropy_rl
```
​
Step 7: Install these packages
```
conda install -c conda-forge libstdcxx-ng=12
sudo apt update
sudo apt-get install patchelf
sudo apt-get install python3-dev build-essential libssl-dev libffi-dev libxml2-dev  
sudo apt-get install libxslt1-dev zlib1g-dev libglew1.5 libglew-dev python3-pip
```
​
Step 8: Install the rest of the packages from [requirements.txt]()
​
```
pip install -r requirements.txt
```
​
Now, you're good to go!


## Run the MultiGaol Environment

Run the following code:

```bash
python ./S2AC/main.py --env Multigoal --max_experiment_steps 5e5 --seed 33 --actor svgd_nonparam --train_action_selection random --test_action_s
election softmax --gpu_id 1 --svgd_steps 10 --a_c 0.2 --a_a 0.2
```

## Run the MuJoCo Experiment

Run the following code:

```bash
python ./S2AC/main.py --env Walker2d-v2 --max_experiment_steps 5e5 --seed 33 --actor svgd_p0_pram --train_action_selection random --test_action_s
election softmax --gpu_id 1 --svgd_steps 3 --a_c 0.2 --a_a 0.2
```


## Run the Toy Experiment

Run the cells in the Notebook ```ToyExperiments.ipynb``` 
