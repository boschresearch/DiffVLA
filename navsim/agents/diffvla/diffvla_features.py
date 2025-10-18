from enum import IntEnum
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import numpy.typing as npt
import json
import os
import pickle
import torch
import sys


from shapely import affinity
from shapely.geometry import Polygon, LineString

from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer, MapObject
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.diffvla.diffvla_config import DiffvlaConfigV2
from navsim.agents.diffvla.bevfusion.transforms_3d import ImageAug3D, ImageNorm
from navsim.common.dataclasses import AgentInput, Scene, Annotations
from navsim.common.enums import BoundingBoxIndex
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder

class DiffvlaFeatureBuilder(AbstractFeatureBuilder):
    """Input feature builder for TransFuser."""

    def __init__(self, config: DiffvlaConfigV2, is_train=True):
        """
        Initializes feature builder.
        :param config: global config dataclass of TransFuser
        """
        self._config = config
        self.image_augmnetor = ImageAug3D(
            final_dim=[256, 704],
            resize_lim=[0.38, 0.55],
            bot_pct_lim=[0.0, 0.0],
            rot_lim=[-5.4, 5.4],
            rand_flip=False,
            is_train=is_train
        )
        self.image_norm = ImageNorm()

        # add vlm cmd
        # self.vlm_data = self._get_vlm_data(config.vlm_json_path)

    # add vlm cmd
    def _get_vlm_data(self, json_path: str) -> List:
        with open(json_path, "r") as f:
            all_data = json.load(f)
        return all_data

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "transfuser_feature"
    
    def validate_projection(self, features, eps: float = 1e-3):
        ori_shapes = features["metas"]["ori_shapes"]
        imgs = features["imgs"] 
        pts = features["points"] # (N, 4)
        pc_xyz = pts[:, 0:3].cpu().numpy() # (N, 3)
        lidar2img = features["metas"]["lidar2img"] # (num_cams, 4, 4)
        num_cams = len(imgs)
        for i in range(num_cams):
            img = imgs[i] # (1080, 1920, 3)
            lidar2img_rt = lidar2img[i] # (4, 4)
            cur_pc_xyz = np.concatenate([pc_xyz, np.ones_like(pc_xyz)[:, :1]], -1) # (N, 4)
            cur_pc_cam = lidar2img_rt @ cur_pc_xyz.T # (4, N)
            cur_pc_cam = cur_pc_cam.T # (N, 4)
            cur_pc_in_fov = cur_pc_cam[:, 2] > eps # (N, 4)
            cur_pc_cam = cur_pc_cam[..., 0:2] / np.maximum(cur_pc_cam[..., 2:3], np.ones_like(cur_pc_cam[..., 2:3]) * eps) # (N, 2)
            
            img_h, img_w = ori_shapes[i]
            cur_pc_in_fov = (
                cur_pc_in_fov
                & (cur_pc_cam[:, 0] < (img_w - 1))
                & (cur_pc_cam[:, 0] > 0)
                & (cur_pc_cam[:, 1] < (img_h - 1))
                & (cur_pc_cam[:, 1] > 0)
            )
            
            for (x, y) in cur_pc_cam[cur_pc_in_fov]: 
                cv2.circle(img, (int(x), int(y)), radius=2, color=(0, 255, 0), thickness=-1)
                
            cv2.imwrite(f"debug_vis/proj_cam_{i}.png", img[:, :, ::-1])
    
    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        features = {}

        # 图像
        imgs, cam2img, lidar2cam, cam2lidar, lidar2img, ori_shapes = self._get_cameras(agent_input)
        features["imgs"] = imgs
        
        features["metas"] = {}
        features["metas"]["ori_shapes"] = np.stack(ori_shapes, axis=0) # (num_cams, 2)
        features["metas"]["cam2img"] = np.stack(cam2img, axis=0) # (num_cams, 4, 4)
        features["metas"]["lidar2cam"] = np.stack(lidar2cam, axis=0)
        features["metas"]["lidar2img"] = np.stack(lidar2img, axis=0)
        features["metas"]["cam2lidar"] = np.stack(cam2lidar, axis=0) # (num_cams, 4, 4)

        #features["metas"]["raw_img"]
        features = self.image_augmnetor(features)
        
        features = self.image_norm(features) # (num_cams, 3, 256, 704)
        
        features["status_feature"] = torch.concatenate(
            [
                torch.tensor(
                    agent_input.ego_statuses[-1].driving_command, dtype=torch.float32
                ),
                torch.tensor(
                    agent_input.ego_statuses[-1].ego_velocity, dtype=torch.float32
                ),
                torch.tensor(
                    agent_input.ego_statuses[-1].ego_acceleration, dtype=torch.float32
                ),
                torch.tensor(
                    agent_input.ego_statuses[-2].driving_command, dtype=torch.float32
                ),
                torch.tensor(
                    agent_input.ego_statuses[-2].ego_velocity, dtype=torch.float32
                ),
                torch.tensor(
                    agent_input.ego_statuses[-2].ego_acceleration, dtype=torch.float32
                ),
                torch.tensor(
                    agent_input.ego_statuses[-3].driving_command, dtype=torch.float32
                ),
                torch.tensor(
                    agent_input.ego_statuses[-3].ego_velocity, dtype=torch.float32
                ),
                torch.tensor(
                    agent_input.ego_statuses[-3].ego_acceleration, dtype=torch.float32
                ),
                torch.tensor(
                    agent_input.ego_statuses[-4].driving_command, dtype=torch.float32
                ),
                torch.tensor(
                    agent_input.ego_statuses[-4].ego_velocity, dtype=torch.float32
                ),
                torch.tensor(
                    agent_input.ego_statuses[-4].ego_acceleration, dtype=torch.float32
                ),
            ],
        )


        return features

    def _get_cameras(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Extract stitched camera from AgentInput
        :param agent_input: input dataclass
        :return: stitched front view image as torch tensor
        """

        cameras = agent_input.cameras[-1]

        imgs, cam2img, lidar2cam, cam2lidar, lidar2img, ori_shapes = [], [], [], [], [], []
        
        # cam_names = ["f0", "l0", "l1", "l2", "r0", "r1", "r2", "b0"]
        cam_names = self._config.cam_names
        
        for cam_name in cam_names:
            
            cam_key = "cam_" + cam_name
            cam = getattr(cameras, cam_key)
            
            img = cv2.undistort(cam.image, cam.intrinsics, cam.distortion)
            imgs.append(img)
            

            intrinsic = cam.intrinsics # (3, 3)
            cam2img_array = np.eye(4).astype(np.float32)
            cam2img_array[:3, :3] = np.array(intrinsic).astype(np.float32)
            cam2img.append(cam2img_array)
            
            cam2lidar_r = cam.sensor2lidar_rotation # (3, 3)
            cam2lidar_t = cam.sensor2lidar_translation # (3,)
            cam2lidar_array = np.eye(4)
            cam2lidar_array[:3, :3] = cam2lidar_r
            cam2lidar_array[:3, 3:4] = cam2lidar_t.reshape(3, 1)
            cam2lidar.append(cam2lidar_array)
            
            lidar2cam_r = np.linalg.inv(cam2lidar_r)
            lidar2cam_t = cam2lidar_t @ lidar2cam_r.T
            lidar2cam_array = np.eye(4)
            lidar2cam_array[:3, :3] = lidar2cam_r.T
            lidar2cam_array[3, :3] = -lidar2cam_t
            lidar2cam.append(lidar2cam_array)
            
            viewpad = np.eye(4)
            viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
            lidar2img_array = viewpad @ lidar2cam_array.T
            lidar2img.append(lidar2img_array)
            
            # shapes
            ori_shape = img.shape[0:2] # (1080, 1920)
            ori_shapes.append(ori_shape)
            
        return imgs, cam2img, lidar2cam, cam2lidar, lidar2img,  ori_shapes

    def _get_lidar(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Compute raw LiDAR pointcloud
        :param agent_input: input dataclass
        :return: pointcloud as torch tensors
        """
        lidar_pc = agent_input.lidars[-1].lidar_pc[0:4].T # (num_points, 4) --> [x, y, z, intensity]
        
        in_range_flags = ((lidar_pc[:, 0] > self._config.lidar_min_x)
                          & (lidar_pc[:, 1] > self._config.lidar_min_y)
                          & (lidar_pc[:, 2] > self._config.lidar_min_z)
                          & (lidar_pc[:, 0] < self._config.lidar_max_x)
                          & (lidar_pc[:, 1] < self._config.lidar_max_y)
                          & (lidar_pc[:, 2] < self._config.lidar_max_z))
        lidar_pc = lidar_pc[in_range_flags]
        return torch.tensor(lidar_pc)


class DiffvlaTargetBuilder(AbstractTargetBuilder):
    """Output target builder for TransFuser."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        config: DiffvlaConfigV2,
    ):
        """
        Initializes target builder.
        :param trajectory_sampling: trajectory sampling specification
        :param config: global config dataclass of Diffvla
        """
        self._trajectory_sampling = trajectory_sampling
        self._config = config

        self.pdm_pkl_path = config.pdm_pkl_path

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "transfuser_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""

        trajectory = torch.tensor(
            scene.get_future_trajectory(
                num_trajectory_frames=self._trajectory_sampling.num_poses
            ).poses
        )
        frame_idx = scene.scene_metadata.num_history_frames - 1
        annotations = scene.frames[frame_idx].annotations
        ego_pose = StateSE2(*scene.frames[frame_idx].ego_status.ego_pose)

        agent_states, agent_labels = self._compute_agent_targets(annotations)
        bev_semantic_map = self._compute_bev_semantic_map(
            annotations, scene.map_api, ego_pose
        )
        
        # for pdm score
         # add pdm scores
        frame_token = scene.frames[frame_idx].token
        pdm_score_path = os.path.join(self.pdm_pkl_path, f"{frame_token}.pkl")
        if os.path.exists(pdm_score_path):
            with open(pdm_score_path, "rb") as f:
                pdm_scores = pickle.load(f)
                nc =  torch.Tensor(pdm_scores['no_at_fault_collisions'])
                dac =  torch.Tensor(pdm_scores['drivable_area_compliance'])
                ddc =  torch.Tensor(pdm_scores['driving_direction_compliance'])
                tlc =  torch.Tensor(pdm_scores['traffic_light_compliance'])
                ep =  torch.Tensor(pdm_scores['ego_progress'])
                tc =  torch.Tensor(pdm_scores['time_to_collision_within_bound'])
                lk =  torch.Tensor(pdm_scores['lane_keeping'])
                hc =  torch.Tensor(pdm_scores['history_comfort'])
        else:
            pdm_scores = None
            raise RuntimeError(f"No PDM score found for token {frame_token} !")

        return {
            "trajectory": trajectory,
            "agent_states": agent_states,
            "agent_labels": agent_labels,
            "bev_semantic_map": bev_semantic_map,
            "nc": nc,
            "dac": dac,
            "ddc": ddc,
            "tlc": tlc,
            "ep": ep,
            "tc": tc,
            "lk": lk,
            "hc": hc,
        }
        
    def debug_bev_semantic_map(
        self, annotations: Annotations, map_api: AbstractMap, ego_pose: StateSE2
    ) -> torch.Tensor:
        """
        Creates sematic map in BEV
        :param annotations: annotation dataclass
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :return: 2D torch tensor of semantic labels
        """
        bev_semantic_map = np.zeros(self._config.bev_semantic_frame, dtype=np.int64)
        for label, (entity_type, layers) in self._config.bev_semantic_classes.items():
            if entity_type == "box":
                box_polygon_mask = np.zeros(
                    self._config.bev_semantic_frame[::-1], dtype=np.uint8
                ) # (256, 256)
                idx = 0
                for name_value, box_value in zip(annotations.names, annotations.boxes):
                    agent_type = tracked_object_types[name_value]
                    if agent_type in layers and name_value == "vehicle":
                        idx += 1
                        # box_value = (x, y, z, length, width, height, yaw) TODO: add intenum
                        x, y, heading = box_value[0], box_value[1], box_value[-1]
                        box_length, box_width, box_height = (
                            box_value[3],
                            box_value[4],
                            box_value[5],
                        )
                        agent_box = OrientedBox(
                            StateSE2(x, y, heading), box_length, box_width, box_height
                        )
                        exterior = np.array(agent_box.geometry.exterior.coords).reshape(
                            (-1, 1, 2)
                        ) # (5, 1, 2)
                        import pdb; pdb.set_trace()
                        exterior = self._coords_to_pixel(exterior)
                        cv2.fillPoly(box_polygon_mask, [exterior], color=255)
                        # 不旋转
                        from navsim.agents.diffvla.transfuser_callback import semantic_map_to_rgb
                        semantic_map = np.zeros(self._config.bev_semantic_frame, dtype=np.int64)
                        semantic_map[box_polygon_mask > 0] = label
                        map_save = semantic_map_to_rgb(semantic_map, self._config)
                        cv2.imwrite("debug_vis/debug_map_%d_1.png"%(idx), map_save)
                        # 旋转
                        semantic_map = np.zeros(self._config.bev_semantic_frame, dtype=np.int64)
                        semantic_map[np.rot90(box_polygon_mask)[::-1] > 0] = label
                        map_save = semantic_map_to_rgb(semantic_map, self._config)
                        cv2.imwrite("debug_vis/debug_map_%d_2.png"%(idx), map_save)
                        # 黑白
                        semantic_map = np.full(self._config.bev_semantic_frame, 255, dtype=np.int64)
                        semantic_map[box_polygon_mask > 0] = 0
                        cv2.imwrite("debug_vis/debug_map_%d_3.png"%(idx), semantic_map)
                        # 黑白
                        semantic_map = np.full(self._config.bev_semantic_frame, 255, dtype=np.int64)
                        semantic_map[np.rot90(box_polygon_mask)[::-1] > 0] = 0
                        cv2.imwrite("debug_vis/debug_map_%d_4.png"%(idx), semantic_map)
                # OpenCV has origin on top-left corner
                box_polygon_mask = np.rot90(box_polygon_mask)[::-1]    
                box_polygon_mask = box_polygon_mask > 0
                bev_semantic_map[box_polygon_mask] = label
        import pdb; pdb.set_trace()
        return torch.Tensor(bev_semantic_map)

    def _compute_agent_targets(
        self, annotations: Annotations
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts 2D agent bounding boxes in ego coordinates
        :param annotations: annotation dataclass
        :return: tuple of bounding box values and labels (binary)
        """

        max_agents = self._config.num_bounding_boxes
        agent_states_list: List[npt.NDArray[np.float32]] = []

        def _xy_in_lidar(x: float, y: float, config: DiffvlaConfigV2) -> bool:
            return (config.lidar_min_x <= x <= config.lidar_max_x) and (
                config.lidar_min_y <= y <= config.lidar_max_y
            )

        for box, name in zip(annotations.boxes, annotations.names):
            box_x, box_y, box_heading, box_length, box_width = (
                box[BoundingBoxIndex.X],
                box[BoundingBoxIndex.Y],
                box[BoundingBoxIndex.HEADING],
                box[BoundingBoxIndex.LENGTH],
                box[BoundingBoxIndex.WIDTH],
            )

            if name == "vehicle" and _xy_in_lidar(box_x, box_y, self._config):
                agent_states_list.append(
                    np.array(
                        [box_x, box_y, box_heading, box_length, box_width],
                        dtype=np.float32,
                    )
                )

        agents_states_arr = np.array(agent_states_list)

        # filter num_instances nearest
        agent_states = np.zeros(
            (max_agents, BoundingBox2DIndex.size()), dtype=np.float32
        )
        agent_labels = np.zeros(max_agents, dtype=bool)

        if len(agents_states_arr) > 0:
            distances = np.linalg.norm(
                agents_states_arr[..., BoundingBox2DIndex.POINT], axis=-1
            )
            argsort = np.argsort(distances)[:max_agents]

            # filter detections
            agents_states_arr = agents_states_arr[argsort]
            agent_states[: len(agents_states_arr)] = agents_states_arr
            agent_labels[: len(agents_states_arr)] = True

        return torch.tensor(agent_states), torch.tensor(agent_labels)

    def _compute_bev_semantic_map(
        self, annotations: Annotations, map_api: AbstractMap, ego_pose: StateSE2
    ) -> torch.Tensor:
        """
        Creates sematic map in BEV
        :param annotations: annotation dataclass
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :return: 2D torch tensor of semantic labels
        """

        bev_semantic_map = np.zeros(self._config.bev_semantic_frame, dtype=np.int64)
        for label, (entity_type, layers) in self._config.bev_semantic_classes.items():
            if entity_type == "polygon":
                entity_mask = self._compute_map_polygon_mask(map_api, ego_pose, layers)
            elif entity_type == "linestring":
                entity_mask = self._compute_map_linestring_mask(
                    map_api, ego_pose, layers
                )
            else:
                entity_mask = self._compute_box_mask(annotations, layers)
            bev_semantic_map[entity_mask] = label

        return torch.Tensor(bev_semantic_map)

    def _compute_map_polygon_mask(
        self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary mask given a map layer class
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: binary mask as numpy array
        """

        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        )
        map_polygon_mask = np.zeros(
            self._config.bev_semantic_frame[::-1], dtype=np.uint8
        )
        for layer in layers:
            for map_object in map_object_dict[layer]:
                polygon: Polygon = self._geometry_local_coords(
                    map_object.polygon, ego_pose
                )
                exterior = np.array(polygon.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(map_polygon_mask, [exterior], color=255)
        # OpenCV has origin on top-left corner
        map_polygon_mask = np.rot90(map_polygon_mask)[::-1]
        return map_polygon_mask > 0

    def _compute_map_linestring_mask(
        self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary of linestring given a map layer class
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: binary mask as numpy array
        """
        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        )
        map_linestring_mask = np.zeros(
            self._config.bev_semantic_frame[::-1], dtype=np.uint8
        )
        for layer in layers:
            for map_object in map_object_dict[layer]:
                linestring: LineString = self._geometry_local_coords(
                    map_object.baseline_path.linestring, ego_pose
                )
                points = np.array(linestring.coords).reshape((-1, 1, 2))
                points = self._coords_to_pixel(points)
                cv2.polylines(
                    map_linestring_mask,
                    [points],
                    isClosed=False,
                    color=255,
                    thickness=2,
                )
        # OpenCV has origin on top-left corner
        map_linestring_mask = np.rot90(map_linestring_mask)[::-1]
        return map_linestring_mask > 0

    def _compute_box_mask(
        self, annotations: Annotations, layers: TrackedObjectType
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary of bounding boxes in BEV space
        :param annotations: annotation dataclass
        :param layers: bounding box labels to include
        :return: binary mask as numpy array
        """
        box_polygon_mask = np.zeros(
            self._config.bev_semantic_frame[::-1], dtype=np.uint8
        )
        for name_value, box_value in zip(annotations.names, annotations.boxes):
            agent_type = tracked_object_types[name_value]
            if agent_type in layers:
                # box_value = (x, y, z, length, width, height, yaw) TODO: add intenum
                x, y, heading = box_value[0], box_value[1], box_value[-1]
                box_length, box_width, box_height = (
                    box_value[3],
                    box_value[4],
                    box_value[5],
                )
                agent_box = OrientedBox(
                    StateSE2(x, y, heading), box_length, box_width, box_height
                )
                exterior = np.array(agent_box.geometry.exterior.coords).reshape(
                    (-1, 1, 2)
                )
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(box_polygon_mask, [exterior], color=255)
        # OpenCV has origin on top-left corner
        box_polygon_mask = np.rot90(box_polygon_mask)[::-1]
        return box_polygon_mask > 0

    @staticmethod
    def _query_map_objects(
        self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> List[MapObject]:
        """
        Queries map objects
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: list of map objects
        """

        # query map api with interesting layers
        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self, layers=layers
        )
        map_objects: List[MapObject] = []
        for layer in layers:
            map_objects += map_object_dict[layer]
        return map_objects

    @staticmethod
    def _geometry_local_coords(geometry: Any, origin: StateSE2) -> Any:
        """
        Transform shapely geometry in local coordinates of origin.
        :param geometry: shapely geometry
        :param origin: pose dataclass
        :return: shapely geometry
        """

        a = np.cos(origin.heading)
        b = np.sin(origin.heading)
        d = -np.sin(origin.heading)
        e = np.cos(origin.heading)
        xoff = -origin.x
        yoff = -origin.y

        translated_geometry = affinity.affine_transform(
            geometry, [1, 0, 0, 1, xoff, yoff]
        )
        rotated_geometry = affinity.affine_transform(
            translated_geometry, [a, b, d, e, 0, 0]
        )

        return rotated_geometry


    def _coords_to_pixel(self, coords):
        """
        Transform local coordinates in pixel indices of BEV map
        :param coords: _description_
        :return: _description_
        """
        # 全景
        pixel_center = np.array([[self._config.bev_pixel_height / 2.0, self._config.bev_pixel_width / 2.0]])
        coords_idcs = (coords / self._config.bev_pixel_size) + pixel_center

        return coords_idcs.astype(np.int32)


class DiffvlaTargetBuilderTest(DiffvlaTargetBuilder):
    """Output target builder for TransFuser."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        config: DiffvlaConfigV2,
    ):
        """
        Initializes target builder.
        :param trajectory_sampling: trajectory sampling specification
        :param config: global config dataclass of TransFuser
        """
        self._trajectory_sampling = trajectory_sampling
        self._config = config

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "transfuser_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        frame_idx = scene.scene_metadata.num_history_frames - 1
        annotations = scene.frames[frame_idx].annotations
        ego_pose = StateSE2(*scene.frames[frame_idx].ego_status.ego_pose)

        agent_states, agent_labels = self._compute_agent_targets(annotations)
        bev_semantic_map = self._compute_bev_semantic_map(
            annotations, scene.map_api, ego_pose
        )
        return {
            "agent_states": agent_states,
            "agent_labels": agent_labels,
            "bev_semantic_map": bev_semantic_map,
        }

class BoundingBox2DIndex(IntEnum):
    """Intenum for bounding boxes in TransFuser."""

    _X = 0
    _Y = 1
    _HEADING = 2
    _LENGTH = 3
    _WIDTH = 4

    @classmethod
    def size(cls):
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_")
            and not attribute.startswith("__")
            and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def LENGTH(cls):
        return cls._LENGTH

    @classmethod
    @property
    def WIDTH(cls):
        return cls._WIDTH

    @classmethod
    @property
    def POINT(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def STATE_SE2(cls):
        # assumes X, Y, HEADING have subsequent indices
        return slice(cls._X, cls._HEADING + 1)
