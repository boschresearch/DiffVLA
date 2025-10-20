# 🧭 How to Use DiffVLA

This guide explains how to set up and configure the **DiffVLA** module inside the NAVSIM framework.

---

## ⚙️ 1. Configuration

Please update the following parameters in  
`navsim/agents/diffvla/transfuser_config.py`:

| Parameter | Description |
|------------|--------------|
| `pdm_file_path` | Path to your PDM (Perception Driving Module) file. |
| `num_voc` | Number of vocabulary entries for language or semantic tokens. |
| `plan_anchor_path` | Path to the anchor points used for trajectory planning. |

**Example:**
```python
pdm_file_path = "/path/to/your/pdm_file.pkl"
num_voc = 64
plan_anchor_path = "/path/to/anchors.json"

## 🚗 2. Trajectory Head
In the new version, the trajectory head has been replaced with a reward-based transformer head.

Please ensure that:

The active trajectory head is switched to
navsim/agents/diffvla/trajectory_head_reward.py

The model definition file
navsim/agents/diffvla/transfuser_model.py
has been updated to use the transformer-based trajectory head.

## 3. Notes

Make sure all paths in configuration files are absolute or relative to the project root.

Cached or generated training data should be excluded via .gitignore.

You can verify your configuration by running:

python navsim/agents/diffvla/validate_config.py

## 4. (Optional) Training and Evaluation

Below is a recommended structure to extend this guide later:

# 🏋️ Training

Run the following command to start training:

bash scripts/train_diffvla.sh


Edit your training script under scripts/ to specify dataset path, GPUs, and hyperparameters.

# 🔍 Evaluation

Run the evaluation script:

bash scripts/eval_diffvla.sh


Evaluation results will be saved in the results/ directory by default.