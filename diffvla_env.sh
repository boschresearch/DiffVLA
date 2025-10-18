#!bin/bash 
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0" 
export NUPLAN_MAPS_ROOT="/data/public/navsim/maps" 
export NAVSIM_EXP_ROOT="/data/gyu3nj/navsim/diffvla_data_exp" 
export NAVSIM_DEVKIT_ROOT="/data/gyu3nj/navsim//navsim" 
export OPENSCENE_DATA_ROOT="/data/public/navsim"

export CUDA_HOME=/usr/local/cuda-12.4
export USE_PYGEOS=0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH