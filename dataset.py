from pathlib import Path
from torch.utils.data import Dataset
from functools import lru_cache

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import trimesh
import albumentations as A
import torch

class ICBINDataset(Dataset):

    cam_params = {
        "cx": 316.0,
        "cy": 244.0,
        "depth_scale": 1.0,
        "fx": 550.0,
        "fy": 540.0,
        "height": 480,
        "width": 640
    }

    WITHIN_MASK = False

    N_PTS = 12
    N_CLASSES = 2

    def __init__(self, 
                 path_to_scenes,
                 path_to_meshes,
                 transform=None):

        self.path_to_scenes = path_to_scenes
        self.path_to_meshes = path_to_meshes
        self.transform = transform
        self.scenes = {}
        
        pose_cfg = "scene_gt.json"
        bbox_cfg = "scene_gt_info.json"
        
        i = 0
        for scene_path in sorted(os.listdir(path_to_scenes)):
            
            rel_scene_path = os.path.join(path_to_scenes, scene_path)
            abs_scene_path = Path(rel_scene_path).absolute()

            if not os.path.isdir(abs_scene_path):
                continue

            images = sorted([f for f in os.listdir(abs_scene_path / "rgb") if f.endswith(".jpg")])
            masks = sorted([f for f in os.listdir(abs_scene_path / "mask") if f.endswith(".png")])

            for image in images:

                # a image consists of a id that follows the following pattern: xxxxxx.jpg
                # a mask consists of a id that follows the following pattern: xxxxxx_yyyyyy.png

                image_id = image.split(".")[0]
                image_masks = [mask for mask in masks if mask.startswith(image_id)]

                for mask in image_masks:

                    self.scenes[i] = {
                        "path": abs_scene_path,
                        "image": image,
                        "mask": mask,
                        "pose_cfg": pose_cfg,
                        "bbox_cfg": bbox_cfg
                    }

                    i += 1
            
            break

        self.K = np.array([[self.cam_params["fx"], 0,                     self.cam_params["cx"]],
                           [0,                     self.cam_params["fy"], self.cam_params["cy"]],
                           [0,                     0,                     1]])

    def __len__(self):

        return len(self.scenes)
    
    @lru_cache(maxsize=32)
    def load_image(self, image_path):
        
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image
    
    @lru_cache(maxsize=32)
    def load_pose_info(self, path_to_pose_info):

        with open(path_to_pose_info, "r") as f:
            pose_cfg = json.load(f)

        return pose_cfg
    
    @lru_cache(maxsize=32)
    def load_bbox_info(self, path_to_bbox_info):

        with open(path_to_bbox_info, "r") as f:
            bbox_cfg = json.load(f)

        return bbox_cfg
    
    def load_mask(self, path_to_mask):

        mask = cv2.imread(str(path_to_mask), cv2.IMREAD_GRAYSCALE)

        return mask
    
    def __getitem__(self, index):                      
        
        scene = self.scenes[index]
        
        # load the rgb image
        image_path = scene["path"] / "rgb" / scene["image"]
        image = self.load_image(image_path)

        # load the pose information
        load_pose_info = scene["path"] / scene["pose_cfg"]
        pose_cfg = self.load_pose_info(load_pose_info)

        # load the bounding box information
        load_bbox_info = scene["path"] / scene["bbox_cfg"]
        bbox_cfg = self.load_bbox_info(load_bbox_info)
        
        # load the mask related to the object
        path_to_mask = scene["path"] / "mask" / scene["mask"]
        mask = self.load_mask(path_to_mask)
        
        global_idx, physical_idx = scene["mask"].split("_")
        global_idx = int(global_idx)
        physical_idx = int(physical_idx.split(".")[0])
        
        # get the bounding box of the object
        x, y, w, h = bbox_cfg[f"{global_idx}"][physical_idx]["bbox_obj"]
        x = max(0, x - int(0.1 * w))
        y = max(0, y - int(0.1 * h))
        w = min(image.shape[1] - x, int(1.2 * w))
        h = min(image.shape[0] - y, int(1.2 * h))

        # crop the image and the mask
        mask = np.zeros((self.N_PTS, self.cam_params["height"], self.cam_params["width"]))
        
        # extract the meta data of the object
        obj = pose_cfg[f"{global_idx}"][physical_idx]
        cam_R_m2c = obj['cam_R_m2c']
        cam_t_m2c = obj['cam_t_m2c']
        obj_id = obj['obj_id']
        
        # load the mesh
        mesh = trimesh.load(f'{self.path_to_meshes}/obj_00000{obj_id}_target.ply')

        # project the mesh into the image
        H = np.eye(4)
        H[:3, :3] = np.array(cam_R_m2c).reshape(3, 3)
        H[:3, 3] = np.array(cam_t_m2c).reshape(3)
        P = np.matmul(self.K, H[:3, :])

        pts = np.array(mesh.vertices)
        pts = np.hstack([pts, np.ones((pts.shape[0], 1))])
        pts = np.matmul(P, pts.T).T
        pts = pts[:, :2] / pts[:, 2:]
        pts = pts.astype(np.int32)

        for j, pt in enumerate(pts):
            
            if pt[0] >= 0 and pt[0] < self.cam_params["width"] and pt[1] >= 0 and pt[1] < self.cam_params["height"]:
                mask[j, pt[1], pt[0]] = 1

        mask = mask[:, y:y+h, x:x+w]

        mask = np.array([cv2.resize(m, (256, 256)) for m in mask])

        mask = np.array([cv2.GaussianBlur(m, (5, 5), 0) for m in mask.astype(np.float32)])
        mask = torch.from_numpy(mask)

        crop = image[y:y+h, x:x+w]
        crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_CUBIC)
        crop = crop.astype(np.float32)
        crop = self.transform(image=crop)["image"]

        crop = np.transpose(crop, (2, 0, 1))
        crop = torch.from_numpy(crop)
        
        crops = torch.stack(crop)
        mask = torch.stack(mask)

        return crops, mask

        
if __name__ == "__main__":

    icbin_dataset = ICBINDataset(path_to_scenes="datasets/icbin",
                                 path_to_meshes="datasets/meshes",
                                 transform=A.Compose([A.Normalize()]))
    icbin_dataset[0]

    
