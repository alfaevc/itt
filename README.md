# Implicit Two-Tower Policies

We present a new class of structured reinforcement learning policy-architectures, Implicit Two-Tower (ITT) policies, where the actions are chosen based on the attention scores of their learnable latent representations with those of the input states. By explicitly disentangling action from state processing in the policy stack, we achieve two main goals: substantial computational gains and better performance. Our architectures are compatible with both: discrete and continuous action spaces. By conducting tests on 15 environments from OpenAI Gym and DeepMind Control Suite, we show that ITT-architectures are particularly suited for blackbox/evolutionary optimization and the corresponding policy training algorithms outperform their vanilla unstructured implicit counterparts as well as commonly used explicit policies. We complement our analysis by showing how techniques such as hashing and lazy tower updates, critically relying on the two-tower structure of ITTs, can be applied to obtain additional computational improvements. 

Paper link: https://arxiv.org/abs/2208.01191


## Installation
The instructions below were tested in Google Cloud Compute with Ubuntu version 18.04.


```
sudo apt-get update
sudo apt-get install python3-pip
sudo apt install unzip gcc libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
sudo ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so
wget https://www.roboti.us/download/mujoco200_linux.zip -P /tmp
unzip /tmp/mujoco200_linux.zip -d ~/.mujoco
mv ~/.mujoco/mujoco200_linux ~/.mujoco/mujoco200
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin" >> ~/.bashrc
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```
Copy the mujoco key to `~/.mujoco/mjkey.txt`, and restart the terminal.

```
conda create --name implicit-two-tower python=3.8
conda activate implicit-two-tower
pip install gym==0.13.1
pip install mujoco-py==2.0.2.10
pip install torch 
pip install pybullet
```

## Running Experiments

The following commands replicate the experiments on HalfCheetah-v2. 

```
python run.py --method='itt' --env='HalfCheetah-v2'
python run.py --method='iot' --env='HalfCheetah-v2'
python run.py --method='explicit' --env='HalfCheetah-v2'
```

To use hashing
```
python run.py --method='itt' --env='HalfCheetah-v2' --n_batch=32 --batch_size=512 --hash=1
```

To use lazy action tower update
```
python run.py --method='itt' --env='HalfCheetah-v2' --lazy=1
```

To use a different environment, for example, Swimmer-v2
```
python run.py --method='itt' --env='Swimmer-v2'
```
In network.py, according to the environment used, add or remove layers in "itt_state_tower", "itt_action_tower", "iot_tower", and "explicit_tower". Update the coorresponding forward function to use desired number of layers and activation function. 