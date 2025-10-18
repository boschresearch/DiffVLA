#!bin/bash 
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0" 
export NUPLAN_MAPS_ROOT="/data/public/navsim/maps" 
export NAVSIM_EXP_ROOT="/data/gyu3nj/navsim/diffvla_data_exp" 
export NAVSIM_DEVKIT_ROOT="/data/gyu3nj/navsim//navsim" 
export OPENSCENE_DATA_ROOT="/data/public/navsim"

CACHE_PATH=$NAVSIM_EXP_ROOT/training_cache_diffvla_6v_pdm8192_vlm

HYDRA_FULL_ERROR=1 python navsim/planning/script/run_dataset_caching.py \
train_test_split=navtrain \
experiment_name=dataset_caching_6v_pdm8192_vlm \
agent=diffvla_agent \
cache_path=$CACHE_PATH \