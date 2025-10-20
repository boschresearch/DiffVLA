CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache_navhard_two_stage

CKPT="exp/diffvla_training_6v_16384pdm_vlm_mask_all_clipe3_clip5_noTTS_FT/2025.09.19.16.32.28/lightning_logs/version_0/checkpoints/epoch-epoch\=14.ckpt"

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