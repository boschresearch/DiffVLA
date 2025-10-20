import pytorch_lightning as pl

from torch import Tensor
from typing import Dict, Tuple

from navsim.agents.abstract_agent import AbstractAgent

from navsim.common.dataclasses import Trajectory

import os
import numpy as np
import pickle


class TrajsAgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent

    def _step(
        self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str
    ) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets = batch
        prediction = self.agent.forward(features, targets)
        loss_dict = self.agent.compute_loss(features, targets, prediction)
        for k, v in loss_dict.items():
            if v is not None:
                self.log(
                    f"{logging_prefix}/{k}",
                    v,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                    batch_size=len(batch[0]),
                )
        return loss_dict["loss"]

    def training_step(
        self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int
    ) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def validation_step(
        self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int
    ):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()

    def test_step(
        self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int
    ):

        features = batch
        try:
            prediction = self.agent.forward(features, targets=None)
        except:
            prediction = self.agent.forward(features)

        log_dir = self.logger.save_dir
        trajs_dir_path = os.path.join(log_dir, "trajs")
        os.makedirs(trajs_dir_path, exist_ok=True)

        trajs_num = prediction["trajectory"].shape[0]
        for i in range(trajs_num):
            token = features["token"][i]
            traj = prediction["trajectory"][i].squeeze(0).cpu().numpy()
            data = Trajectory(traj, self.agent._trajectory_sampling)
            with open(f"{trajs_dir_path}/{token}.pkl", "wb") as f:
                pickle.dump(data, f)

        return prediction
