import logging
from pathlib import Path
from typing import Tuple

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.testing.trajs_agent_lightning_module import (
    TrajsAgentLightningModule,
)
from navsim.planning.testing.dataset import TestCacheOnlyDataset, BEVTestDataset

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def build_datasets(
    cfg: DictConfig, agent: AbstractAgent
) -> Tuple[BEVTestDataset, BEVTestDataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [
            log_name
            for log_name in val_scene_filter.log_names
            if log_name in cfg.test_logs
        ]
    else:
        val_scene_filter.log_names = cfg.test_logs

    data_path = Path(cfg.navsim_log_path)
    original_sensor_path = Path(cfg.original_sensor_path)
    synthetic_sensor_path = Path(cfg.synthetic_sensor_path)
    synthetic_scenes_path = Path(cfg.synthetic_scenes_path)

    val_scene_loader = SceneLoader(
        original_sensor_path=original_sensor_path,
        synthetic_scenes_path=synthetic_scenes_path,
        synthetic_sensor_path=synthetic_sensor_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_data = BEVTestDataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders_test(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
        requires_scene=agent.requires_scene,
    )

    return val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

    logger.info("Building Lightning Module")
    lightning_module = TrajsAgentLightningModule.load_from_checkpoint(
        agent=agent, checkpoint_path=cfg.agent.checkpoint_path, strict=False
    )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        val_data = TestCacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders_test(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.test_logs,
        )
    else:
        logger.info("Building SceneLoader")
        val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    val_dataloader = DataLoader(
        val_data,
        **cfg.dataloader.params,
        shuffle=False,
        collate_fn=val_data.collate_batch,
    )
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    trainer = pl.Trainer(**cfg.trainer.params)

    logger.info("Starting Testing")
    trainer.test(
        model=lightning_module,
        dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    main()
