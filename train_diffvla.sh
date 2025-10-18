#!bin/bash 
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0" 
export NUPLAN_MAPS_ROOT="/data/public/navsim/maps" 
export NAVSIM_EXP_ROOT="/data/gyu3nj/navsim/diffvla_data_exp" 
export NAVSIM_DEVKIT_ROOT="/data/gyu3nj/navsim//navsim" 
export OPENSCENE_DATA_ROOT="/data/public/navsim"

CACHE_PATH=$NAVSIM_EXP_ROOT/training_cache_diffvla_6v_pdm8192

HYDRA_FULL_ERROR=1 python navsim/planning/script/run_training_diffvla.py \
    agent=diffvla_agent \
    experiment_name=diffvla_training_6v_8192pdm \
    train_test_split=navtrain \
    trainer.params.max_epochs=30 \
    split=trainval dataloader.params.batch_size=2 \
    use_cache_without_dataset=True \
    force_cache_computation=False \
    cache_path=$CACHE_PATH \
    agent.lr=1e-4