TRAIN_TEST_SPLIT=navtrain
CHECKPOINT=/data/gyu3nj/navsim_6/navsim/scripts/misc/model.ckpt
CACHE_PATH=/data/gyu3nj/navsim_6/exp/metric_cache_navtrain
SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles

HYDRA_FULL_ERROR=1 python /misc/gen_multi_trajs_pdm_score_ours.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=diffvla_agent \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=full_model_vlm_pdm_score \
metric_cache_path=$CACHE_PATH \
synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
synthetic_scenes_path=$SYNTHETIC_SCENES_PATH \