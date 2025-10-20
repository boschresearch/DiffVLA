#!bin/bash 
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0" 
export NUPLAN_MAPS_ROOT="/data/public/navsim/maps" 
export NAVSIM_EXP_ROOT="/data/gyu3nj/navsim/diffvla_data_exp" 
export NAVSIM_DEVKIT_ROOT="/data/gyu3nj/navsim//navsim" 
export OPENSCENE_DATA_ROOT="/data/public/navsim"

TRAIN_TEST_SPLIT=navhard_two_stage
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache_navhard_two_stage

python $NAVSIM_DEVKIT_ROOT/planning/script/run_metric_caching.py \
train_test_split=$TRAIN_TEST_SPLIT \
metric_cache_path=$CACHE_PATH