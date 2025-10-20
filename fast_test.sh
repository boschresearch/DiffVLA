#!bin/bash 
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0" 
export NUPLAN_MAPS_ROOT="/data/public/navsim/maps" 
export NAVSIM_EXP_ROOT="/data/gyu3nj/navsim/diffvla_data_exp" 
export NAVSIM_DEVKIT_ROOT="/data/gyu3nj/navsim//navsim" 
export OPENSCENE_DATA_ROOT="/data/public/navsim"

CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache_navhard_two_stage

CKPT="diffvla_data_exp/diffvla_training_6v_8192pdm_vlm/2025.10.20.10.54.39/lightning_logs/version_0/checkpoints/epoch-epoch\=4.ckpt"

EXP_NAME="diffvla_8192"
EXP_ID="diffvla_8192_fast_test"
AGENT="diffvla_agent"

CUDA_VISIBLE_DEVICES=0 OPENBLAS_CORETYPE=haswell HYDRA_FULL_ERROR=1 python navsim/planning/script/generate_bev_trajs.py \
    agent=$AGENT \
    experiment_name=$EXP_NAME \
    experiment_uid=$EXP_ID \
    train_test_split=navhard_two_stage \
    dataloader.params.batch_size=2 \
    use_cache_without_dataset=False \
    force_cache_computation=False \
    cache_path=null \
    agent.checkpoint_path=$CKPT

OPENBLAS_CORETYPE=haswell HYDRA_FULL_ERROR=1 python navsim/planning/script/run_pdm_score_trajs.py \
    agent=$AGENT \
    experiment_name=$EXP_NAME \
    experiment_uid=$EXP_ID \
    train_test_split=navhard_two_stage \
    metric_cache_path=$CACHE_PATH