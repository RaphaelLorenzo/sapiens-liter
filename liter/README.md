
<p align="center">
  <img src="../assets/sapiens_liter_logo.png" alt="Sapiens-Liter" title="Sapiens-Liter" width="500"/>
</p>

# Installation

```
conda create --name sapiens_liter python=3.10
conda activate sapiens_liter
pip install torch torchvision torchaudio opencv-python tqdm json-tricks ultralytics 
```

Download the checkpoints and set the path :

```
export SAPIENS_LITE_CHECKPOINT_ROOT=/home/yourself/SAPIENS_CHECKPOINT_DIR/
```

# Pose estimation

Will run with the test images and the default 0.3b model.

```
python run_pose.py
```