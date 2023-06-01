import cv2
import json
import numpy as np
import os
import random
import torch

from collections import defaultdict
from scipy.spatial.transform import Rotation as ScipyRotation
from scipy.spatial.transform import Slerp
from torch.utils.data import Dataset

try:
    from .helper import AugmentationList
except Exception:
    from helper import AugmentationList

from .ngp_utils import get_rays, nerf_matrix_to_ngp

__all__ = ["ScanNetNGPJoint"]


class ScanNetNGPJoint(Dataset):

    def __init__(
        self,
        root,
        scene_list,
        mode="train",
        output_size=(240, 320),
        degrees=10,
        flip_p=0.5,
        jitter_bcsh=[0.3, 0.3, 0.3, 0.05],
        data_augmentation=True,
        exp_name="debug",
        use_novel_viewpoints=False,
        only_new_scene=True,
        fix_nerf=False,
        replay_buffer_size=None,
    ):

        super(ScanNetNGPJoint, self).__init__()
        self._mode = mode

        self.H = output_size[0]
        self.W = output_size[1]

        self.num_rays = 4096

        self.root = root
        self.exp_name = exp_name
        self.fix_nerf = fix_nerf
        # only new scene:
        if only_new_scene:
            scene_list = [scene_list[-1]]

        self.replay_buffer_size = replay_buffer_size
        # deal with replay
        self.replay_per_scene = None
        if self.replay_buffer_size is not None:
            num_old_scenes = len(scene_list) - 1
            if num_old_scenes > 0:
                self.replay_per_scene = (self.replay_buffer_size //
                                         num_old_scenes)

        ## hardcoded
        if self._mode == "val":
            scene_list = [
                "scene0000_00",
                "scene0001_00",
                "scene0002_00",
                "scene0003_00",
                "scene0004_00",
                "scene0005_00",
                "scene0006_00",
                "scene0007_00",
                "scene0008_00",
                "scene0009_00",
            ]

        if self._mode == "train_val":
            scene_list = [
                "scene0000_00",
                "scene0001_00",
                "scene0002_00",
                "scene0003_00",
                "scene0004_00",
                "scene0005_00",
                "scene0006_00",
                "scene0007_00",
                "scene0008_00",
                "scene0009_00",
            ]

        if self._mode == "predict":
            self._use_novel_viewpoints = use_novel_viewpoints
        elif self._mode == "train":
            self._use_novel_viewpoints = (use_novel_viewpoints and
                                          self.replay_per_scene is not None)
        else:
            assert not use_novel_viewpoints
            self._use_novel_viewpoints = False
        self.get_ngp_info(scene_list)

        if self._use_novel_viewpoints:
            self.length = len(self.nerf_image_pths)
        else:
            self.length = len(self.image_pths)
        self._augmenter = AugmentationList(output_size, degrees, flip_p,
                                           jitter_bcsh)
        self._data_augmentation = data_augmentation

    def get_ngp_info(self, scene_list):
        # old scenes:

        self.poses = []
        self.image_pths = []
        self.label_pths = []  # GT
        self.nerf_label_pths = []  # PL of old scenes
        self.nerf_image_pths = []
        self.depth_pths = []
        self.from_old_scene = []  # bool array
        self.viewpoint_is_novel = []
        for i in range(len(scene_list)):
            scene_name = scene_list[i]
            scene_root = os.path.join(self.root, scene_name)
            with open(os.path.join(scene_root, "transforms_train.json"),
                      "r") as f:
                ngp_info = json.load(f)
            # new scene
            if i == len(scene_list) - 1:
                self.ngp_H = int(ngp_info["h"])
                self.ngp_W = int(ngp_info["w"])
                self.ngp_fl_x = ngp_info["fl_x"]
                self.ngp_fl_y = ngp_info["fl_y"]
                self.ngp_cx = ngp_info["cx"]
                self.ngp_cy = ngp_info["cy"]
                self.one_m_to_scene_uom = ngp_info["one_m_to_scene_uom"]
                self.ngp_intrinsics = np.array(
                    [self.ngp_fl_x, self.ngp_fl_y, self.ngp_cx, self.ngp_cy])

            frames = ngp_info["frames"]
            if self._mode != "predict":
                if self._mode == "val":
                    frames = frames[-int(0.2 * len(frames)):]
                    # frames = frames[:len(frames)//4*4]
                else:
                    frames = frames[:-int(0.2 * len(frames))]
            generated_frame_json_path = os.path.join(
                scene_root,
                self.exp_name,
                "novel_viewpoints",
                "interpolated_data.json",
            )
            # if training and replay and from the old scene
            if (self._mode == "train" and self.replay_per_scene is not None and
                (i < len(scene_list) - 1)):
                if self._use_novel_viewpoints:
                    # Read the previously-generated data.
                    with open(generated_frame_json_path, "r") as f:
                        frames = json.load(f)["frames"]
                random.Random(0).shuffle(frames)
                frames = frames[:self.replay_per_scene]

            current_poses = []
            generated_image_paths = []
            generated_label_paths = []

            for f in frames:
                if (self._mode == "train" and
                        self.replay_per_scene is not None and
                        self._use_novel_viewpoints and i < len(scene_list) - 1):
                    # Reading data from novel viewpoints from a previous scene.
                    nerf_image_path = f["nerf_image"]
                    nerf_label_path = f["nerf_label"]
                    pose = np.array(f["pose"], dtype=np.float32)
                    print("\033[96mReading data from replay. Image path = "
                          f"{nerf_image_path}.\033[0m")
                else:
                    image_path = os.path.join(scene_root, f["file_path"])
                    label_path = os.path.join(scene_root, f["label_path"])
                    depth_path = os.path.join(
                        scene_root,
                        "depth",
                        os.path.basename(image_path).split(".")[0] + ".png",
                    )
                    nerf_label_path = os.path.join(
                        scene_root,
                        self.exp_name,
                        "novel_viewpoints"
                        if self._use_novel_viewpoints else "",
                        "nerf_label",
                        os.path.basename(image_path).split(".")[0] + ".png",
                    )
                    nerf_image_path = os.path.join(
                        scene_root,
                        self.exp_name,
                        "novel_viewpoints"
                        if self._use_novel_viewpoints else "",
                        "nerf_image",
                        os.path.basename(image_path).split(".")[0] + ".png",
                    )
                    generated_label_paths.append(nerf_label_path)
                    generated_image_paths.append(nerf_image_path)
                    pose = np.array(f["transform_matrix"],
                                    dtype=np.float32)  # [4, 4]
                current_poses.append(pose)
                if self._use_novel_viewpoints and (
                    (self._mode == "train" and self.replay_per_scene is not None
                     and i < len(scene_list) - 1) or self._mode == "predict"):
                    self.viewpoint_is_novel.append(True)
                    self.image_pths.append(None)
                    self.label_pths.append(None)
                    self.depth_pths.append(None)
                else:
                    self.viewpoint_is_novel.append(False)
                    self.image_pths.append(image_path)
                    self.label_pths.append(label_path)
                    self.depth_pths.append(depth_path)
                self.nerf_label_pths.append(nerf_label_path)
                self.nerf_image_pths.append(nerf_image_path)
                if self._mode == "val" or self._mode == "train_val":
                    self.from_old_scene.append(False)
                elif i < len(scene_list) - 1 or self.fix_nerf:
                    self.from_old_scene.append(True)
                else:
                    self.from_old_scene.append(False)

            if self._use_novel_viewpoints and self._mode == "predict":
                # To generate novel viewpoints, interpolate between the current
                # poses.
                # - Add again first view to allow interpolation between the last
                #   and the first view.
                current_poses.append(current_poses[0])
                # - Associates fake, uniform times to the poses, so that one can
                #   easily interpolate between them.
                fake_times = [*range(len(current_poses))]
                fake_times_interpolated_poses = [
                    0.5 + idx for idx in range(len(current_poses) - 1)
                ]
                # - Interpolate rotations.
                spherical_interpolator = Slerp(
                    times=fake_times,
                    rotations=ScipyRotation.from_matrix(
                        [pose[:3, :3] for pose in current_poses]),
                )
                interpolated_rotations = list(
                    spherical_interpolator(
                        times=fake_times_interpolated_poses).as_matrix())
                # - Interpolate translations.
                interpolated_poses = []
                for pose_idx in range(len(current_poses) - 1):
                    curr_interpolated_pose = np.eye(4)
                    curr_interpolated_pose[:3, :3] = interpolated_rotations[
                        pose_idx]
                    curr_interpolated_pose[:3, 3] = (
                        current_poses[pose_idx][:3, 3] +
                        current_poses[pose_idx + 1][:3, 3]) / 2.0
                    interpolated_poses.append(curr_interpolated_pose)

                assert len(interpolated_poses) == len(current_poses) - 1
                current_poses = interpolated_poses

                assert (len(generated_image_paths) == len(generated_label_paths)
                        == len(current_poses))

                # - Store the information about the new data in a json file.
                generated_frames = []
                for (
                        generated_image_path,
                        generated_label_path,
                        generated_pose_path,
                ) in zip(generated_image_paths, generated_label_paths,
                         current_poses):
                    generated_frames.append({
                        "nerf_image": generated_image_path,
                        "nerf_label": generated_label_path,
                        "pose": generated_pose_path.tolist(),
                    })
                os.makedirs(os.path.dirname(generated_frame_json_path),
                            exist_ok=True)

                # Create JSON file containing the paths to the rendered frames
                # that will be created, as well as the associated (interpolated)
                with open(generated_frame_json_path, "w") as f:
                    json.dump({"frames": generated_frames}, f, indent=5)

            current_poses = [nerf_matrix_to_ngp(p) for p in current_poses]
            self.poses = self.poses + current_poses

        self.poses = torch.from_numpy(np.stack(self.poses, axis=0))

    def preprocess_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = (torch.from_numpy(img).permute(2, 0, 1).type(torch.float32)
              )  # C H W range 0-1
        return img

    def preprocess_label(self, label_path):
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        label = cv2.resize(label, (self.W, self.H),
                           interpolation=cv2.INTER_NEAREST)
        label = torch.from_numpy(label).type(torch.int64)
        label = label - 1  # -1 unknown, 0-39->nyu40
        return label

    def preprocess_depth(self, depth_path):
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = cv2.resize(depth, (self.W, self.H),
                           interpolation=cv2.INTER_NEAREST)
        assert depth.ndim == 2
        assert depth.dtype == np.uint16
        depth = depth.astype(np.float32) / 1000.0
        depth = torch.from_numpy(depth).type(torch.float32)
        return depth

    def __getitem__(self, index):
        # old scene
        if self.from_old_scene[index]:
            nerf_label = self.preprocess_label(self.nerf_label_pths[index])
            nerf_image = self.preprocess_image(self.nerf_image_pths[index])
            if self.viewpoint_is_novel[index]:
                print("\033[96mMixing novel-viewpoint image in current batch. "
                      "The path to the NeRF-rendered image is "
                      f"{self.nerf_image_pths[index]}. The path to the "
                      f"NeRF-rendered label is {self.nerf_label_pths[index]}."
                      "\033[0m")
                img = nerf_image
                img_fp16 = None
                # This to allow easy handling of augmentation. This label should
                # be ignored and is set to `None` later in the code.
                label = nerf_label
                depth = None
            else:
                img = self.preprocess_image(
                    self.image_pths[index]
                    # the img used for inference deeplab should have fp32
                    # precision
                )
                img_fp16 = self.preprocess_image(self.image_pths[index]).type(
                    torch.half)
                label = self.preprocess_label(self.label_pths[index])
                depth = self.preprocess_depth(self.depth_pths[index]).type(
                    torch.half)
            if self._mode == "train" and self._data_augmentation:
                # use nerf image during training
                img, labels = self._augmenter.apply(
                    nerf_image,
                    [
                        (label[None, ...] + 1).type(torch.float32),
                        (nerf_label[None, ...] + 1).type(torch.float32),
                    ],
                )
            else:
                img, labels = self._augmenter.apply(
                    img,
                    [
                        (label[None, ...] + 1).type(torch.float32),
                        (nerf_label[None, ...] + 1).type(torch.float32),
                    ],
                    only_crop=True,
                )

            label, nerf_label = labels
            label = (label[0, ...] - 1).type(
                torch.int64)  # 0 unknown -> -1 unknown
            if self.viewpoint_is_novel[index]:
                # If the viewpoint is novel, the ground-truth label is not
                # available.
                label = None
            nerf_label = (nerf_label[0, ...] - 1).type(torch.int64)

            pose = self.poses[-1].unsqueeze(0)
            rays_test = get_rays(pose, self.ngp_intrinsics, self.ngp_H,
                                 self.ngp_W)
            ret_dict = {
                "img": img,
                "label": label,
                "depth": depth,
                "img_fp16": img_fp16,
                "nerf_label": nerf_label,
                "pose": pose[0],
                "from_old_scene": True,
                "viewpoint_is_novel": self.viewpoint_is_novel[index],
            }
            ret_dict.update({
                "H": self.ngp_H,
                "W": self.ngp_W,
                "intrinsics": self.ngp_intrinsics,
                "one_m_to_scene_uom": self.one_m_to_scene_uom,
                "rays_o": rays_test["rays_o"][0],
                "rays_d": rays_test["rays_d"][0],
                "direction_norms": rays_test["direction_norms"][0],
            })

        else:
            if self.viewpoint_is_novel[index]:
                # In here when generating a novel view during prediction.
                img = []
                img_fp16 = []
                label = []
                depth = []
            else:
                # Read Image and Label
                img = self.preprocess_image(
                    # the img used for inference deeplab should have fp32
                    # precision
                    self.image_pths[index])
                img_fp16 = self.preprocess_image(self.image_pths[index]).type(
                    torch.half)
                label = self.preprocess_label(self.label_pths[index])
                depth = self.preprocess_depth(self.depth_pths[index]).type(
                    torch.half)
            pose = self.poses[index].unsqueeze(0)
            rays_test = get_rays(pose, self.ngp_intrinsics, self.ngp_H,
                                 self.ngp_W)
            ret_dict = {
                "img": img,
                "label": label,
                "depth": depth,
                "img_fp16": img_fp16,
                "nerf_label": label,
                "pose": pose[0],
                "from_old_scene": False,
                "viewpoint_is_novel": self.viewpoint_is_novel[index],
            }
            ret_dict.update({
                "H": self.ngp_H,
                "W": self.ngp_W,
                "intrinsics": self.ngp_intrinsics,
                "one_m_to_scene_uom": self.one_m_to_scene_uom,
                "rays_o": rays_test["rays_o"][0],
                "rays_d": rays_test["rays_d"][0],
                "direction_norms": rays_test["direction_norms"][0],
            })
        # if self._mode == "val":
        # hard coded
        if self.viewpoint_is_novel[index]:
            import re

            current_scene_name = re.findall("scene\d\d\d\d_\d\d",
                                            self.nerf_image_pths[index])
            assert len(current_scene_name) == 1
            current_scene_name = current_scene_name[0]
            current_index = str(
                os.path.basename(self.nerf_image_pths[index])[:-4])
        else:
            current_scene_name = os.path.normpath(self.image_pths[index]).split(
                os.path.sep)[-3]
            # if self._mode == "predict":
            current_index = str(os.path.basename(self.image_pths[index])[:-4])
        ret_dict["current_scene_name"] = current_scene_name
        ret_dict["current_index"] = current_index

        return ret_dict

    @staticmethod
    def collate(batch):
        batch_old = defaultdict(list)
        batch_new = defaultdict(list)
        batch_cl = defaultdict(list)
        for key in batch[0]:

            for i in range(len(batch)):
                if key in ["replay_img", "replay_label"]:
                    batch_cl[key].append(batch[i][key])
                elif batch[i]["from_old_scene"]:
                    batch_old[key].append(batch[i][key])
                else:
                    batch_new[key].append(batch[i][key])
        if not "img" in batch_old:
            batch_old = None
        else:
            for key in batch_old:
                if type(batch_old[key][0]) == torch.Tensor:
                    batch_old[key] = torch.stack(batch_old[key], dim=0)

        if not "img" in batch_new:
            batch_new = None
        else:
            for key in batch_new:
                if type(batch_new[key][0]) == torch.Tensor:
                    batch_new[key] = torch.stack(batch_new[key], dim=0)

        if not "replay_img" in batch_cl:
            batch_cl = None
        else:
            for key in batch_cl:
                if type(batch_cl[key][0]) == torch.Tensor:
                    batch_cl[key] = torch.stack(batch_cl[key], dim=0)

        return batch_old, batch_new, batch_cl

    def __len__(self):
        return self.length

    def __str__(self):
        string = "=" * 90
        string += "\nScannet Dataset: \n"
        length = len(self)
        string += f"    Total Samples: {length}"
        string += f"  »  Mode: {self._mode} \n"
        string += f"  »  DataAug: {self._data_augmentation}"
        string += "=" * 90
        return string
