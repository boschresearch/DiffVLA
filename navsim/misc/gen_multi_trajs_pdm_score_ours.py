import pandas as pd
from tqdm import tqdm
import traceback
import numpy as np

import hydra
from hydra.utils import instantiate


from omegaconf import DictConfig

from pathlib import Path
from typing import Any, Dict, List, Union, Tuple
from dataclasses import asdict
from datetime import datetime
import logging
import lzma
import pickle
import os
import uuid

from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.common.dataloader import MetricCacheLoader
from navsim.agents.abstract_agent import AbstractAgent
from navsim.evaluate.pdm_score import pdm_score, pdm_score_multi_trajs
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator
)
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.common.dataloader import SceneLoader, SceneFilter
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataclasses import AgentInput, Trajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from pathlib import Path


logger = logging.getLogger(__name__)

num_clusters = 512
num_horizons = 8
compute_state_only = False
if compute_state_only:
    print("Computing state only")

save_pkl_path = f'/data/gyu3nj/CVPR2025navsim_6/pdm_scores_{num_clusters}/'
output_dir = Path(save_pkl_path)
output_dir.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = "../../navsim/planning/script/config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"
@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    build_logger(cfg)
    worker = build_worker(cfg)
 
    scene_loader = SceneLoader(
        original_sensor_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    full_tokens_to_evaluate = sorted(list(set(scene_loader.tokens) & set(metric_cache_loader.tokens)))

    if num_splits > 1:
        if split_id < 0 or split_id >= num_splits:
            raise ValueError(f"split_id {split_id} must be between 0 and {num_splits-1}")
        tokens_to_evaluate = full_tokens_to_evaluate[split_id::num_splits]
        logger.info(f"Processing split {split_id}/{num_splits} with {len(tokens_to_evaluate)} tokens (out of {len(full_tokens_to_evaluate)} total).")
    else:
        tokens_to_evaluate = full_tokens_to_evaluate
        logger.info(f"No splitting enabled; processing all {len(tokens_to_evaluate)} tokens.")

    num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
    num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
    if num_missing_metric_cache_tokens > 0:
        logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
    if num_unused_metric_cache_tokens > 0:
        logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    logger.info("Starting pdm scoring of %s scenarios...", str(len(tokens_to_evaluate)))
    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]

    single_eval = getattr(cfg, 'single_eval', False)
    # single-threaded worker_map
    if single_eval:
        print("Running single-threaded worker_map")
        score_rows = run_pdm_score(data_points)
    else:
        # mutli-threaded worker_map
        score_rows: List[Tuple[Dict[str, Any], int, int]] = worker_map(worker, run_pdm_score, data_points)

    # Call the refactored function

def format_and_save_scores(score_rows, num_clusters, output_dir):
    """
    Formats the score rows into a dictionary, saves them, and outputs a summary DataFrame.

    Parameters:
    - score_rows: List of score rows to format and save.
    - num_clusters: Number of clusters to use in the file name.
    - output_dir: Directory to save the output CSV.

    Returns:
    - pd.DataFrame: The formatted DataFrame of scores.
    """
    # Format score_rows into dictionary
    score_dict = {}
    for row in tqdm(score_rows):
        key = row['token']
        value = {k: v for k, v in row.items() if k != 'token'}
        score_dict[key] = value

    # Save formatted score_rows using numpy
    save_path = f'./planning_vb/formatted_pdm_score_{num_clusters}.npy'
    np.save(save_path, score_dict, allow_pickle=True)

    print(f'Saved formatted scores to {save_path}')

    pdm_score_df = pd.DataFrame(score_rows)
    num_successful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_successful_scenarios
    average_row = pdm_score_df.drop(columns=["token", "valid"]).mean(skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = pdm_score_df["valid"].all()
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pdm_score_df.to_csv(f"./planning_vb/pdm_score_df_{timestamp}.csv")

    logger.info(f"""
        Finished running evaluation.
            Number of successful scenarios: {num_successful_scenarios}. 
            Number of failed scenarios: {num_failed_scenarios}.
            Final average score of valid results: {pdm_score_df['score'].mean()}.
            Results are stored in: ./planning_vb/pdm_score_df_{timestamp}.csv
    """)

    return pdm_score_df

def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Dict[str, Any]]:
    """
    Computes PDM scores for a list of tokens using predefined trajectories.
    
    Args:
        args: List of dictionaries containing configuration, log file, and tokens.
    
    Returns:
        List of dictionaries with token scores and metadata.
    """

    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)

    assert simulator.proposal_sampling == scorer.proposal_sampling, "Simulator and scorer proposal sampling has to be identical"
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )

    predefined_trajectories = load_predefined_trajectories()  # Assuming this function is defined to load trajectories

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    pdm_results: List[Dict[str, Any]] = []
    for idx, (token) in tqdm(enumerate(tokens_to_evaluate)):
        score_row: Dict[str, Any] = {"token": token, "valid": True}
        # try:
        metric_cache_path = metric_cache_loader.metric_cache_paths[token]
        with lzma.open(metric_cache_path, "rb") as f:
            metric_cache: MetricCache = pickle.load(f)

        # put new traj code here
        # Load 256 pre-defined trajectories from a file or other source
        trajectory_scores = []

        pdm_result = pdm_score_multi_trajs(
            metric_cache=metric_cache,
            model_trajectory_list=predefined_trajectories,
            # future_sampling=proposal_sampling,
            future_sampling=simulator.proposal_sampling,
            simulator=simulator,
            scorer=scorer,
        )

        pdm_pkl_file_name = f'{save_pkl_path}{token}.pkl'
        with open(pdm_pkl_file_name, 'wb') as f:
            pickle.dump(asdict(pdm_result), f)

        if compute_state_only:
            trajectory_scores.append(pdm_result)
        else:
            trajectory_scores.append(asdict(pdm_result))
        
        # Update the score_row with the computed scores
        score_row["trajectory_scores"] = trajectory_scores

        pdm_results.append(score_row)
    return pdm_results

def load_predefined_trajectories() -> List[Any]:
    """
    Load 256 pre-defined trajectories from the given path.
    Assumes that the trajectories are stored in a serialized format (e.g., pickle).
    """

    path = f'/data/gyu3nj/CVPR2025/navsim_6/navsim/planning_vb/trajectory_anchors_{num_clusters}.npy'
    
    with open(path, "rb") as f:
        trajectories = np.load(f)
    return trajectories

if __name__ == "__main__":
    main()