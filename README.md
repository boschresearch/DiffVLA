<div align="center">
  <img src="assets/diffvla_overview.png" width="420">
  <img src="assets/diffvla++_logo.png" width="420">

  <h2 align="center">DiffVLA: Vision-Language Guided Diffusion Planning for Autonomous Driving</h2>

  <h3 align="center">
    <a href="https://arxiv.org/abs/2505.19381">[Paper]</a> |
    <a href="https://github.com/autonomousvision/navsim">[Base: NAVSIM Repository]</a>
  </h3>

  <p align="center">
    DiffVLA builds upon <a href="https://github.com/autonomousvision/navsim">NAVSIM</a> to enable
    diffusion-driven end-to-end policy learning for language-conditioned autonomous driving.
    <br/>
    This repository integrates pseudo-simulation from NAVSIM with diffusion-based trajectory generation.
  </p>
</div>

<br/>

---

## 📘 Documentation

# 🧭 How to Use DiffVLA
This guide explains how to set up and configure the **DiffVLA** module inside the NAVSIM framework.
---

## ⚙️ 1. Configuration 
Please update the following parameters in

`navsim/agents/diffvla/diffvla_config.py`:


| Parameter | Description |
|------------|--------------|

|  `pdm_pkl_path`  | Path to your PDM GT files. |

|  `num_voc`  | Different number of trajectory vocabs. |

|  `plan_anchor_path`  | Path to the anchor files used for trajectory planning. |

|  `with_vlm`  | Define with or without meta action cmd from VLM. |

**Example:**

pdm_file_path = "diffvla_data_exp/pdm_scores_8192"
num_voc = 8192
plan_anchor_path = "diffvla_data_exp/planning_vb/trajectory_anchors_8192.npy"
with_vlm = True


## 🚗 2. Trajectory Head
In the new version, the trajectory head has been replaced with a reward-based transformer head.

Please ensure that:

The active trajectory head is switched to
navsim/agents/diffvla/trajectory_head_reward.py

The model definition file
navsim/agents/diffvla/transfuser_model.py
has been updated to use the transformer-based trajectory head.


## 🧩 3. Notes
Make sure all paths in configuration files are absolute or relative to the project root.

Cached or generated training data should be excluded via .gitignore.

You can verify your configuration by running:

python navsim/agents/diffvla/validate_config.py

## 🧪 4. (Optional) Training and Evaluation
Below is a recommended structure to extend this guide later:


# 🏋️ Training
Run the following command to define env, cache dataset and start training:

bash diffvla_env.sh 
bash cache_dataset_diffvla.sh
bash train_diffvla.sh

Edit train_diffvla.sh  to specify dataset cache path, GPUs, and hyperparameters.


# 🔍 Evaluation
We offer a fast evaluation script to test the model duo to the original test scripts from NAVSIM are too slow.
 
Run the evaluation script:

bash fast_test.sh

Edit fast_test.sh to the specific CACHE_PATH, CKPT, EXP_NAME, EXP_ID and AGENT 

Evaluation results will be saved in the results/ directory by default.