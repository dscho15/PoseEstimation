from pathlib import Path
from torch.utils.data import Dataset

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import trimesh
import albumentations as A
import torch

class ICBINDataset(Dataset):

    CAM_PARAMS = {
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

    def __init__(self, 
                 path, 
                 transform=None):

        self.path = path
        self.transform = transform
        self.scenes = {}
        
        i = 0
        for scene in os.listdir(path):
            
            scene_path = os.path.join(path, scene)

            if not os.path.isdir(scene_path):
                continue
            
            path_scene = Path(scene_path).absolute()
            images = sorted([f for f in os.listdir(path_scene / "rgb") if f.endswith(".png") or f.endswith(".jpg")])
            masks = sorted([f for f in os.listdir(path_scene / "mask_crushed") if f.endswith(".png")])
            scene_camera = path_scene / "scene_camera.json"

            for img, mask in zip(images, masks):
                idx = int(str(img)[:-4])
                self.scenes[idx] = {
                    "path": path_scene,
                    "rgb": img,
                    "mask": mask,
                    "scene_camera": scene_camera
                }
                i += 1
        
        # camera matrix
        self.K = np.array([[self.CAM_PARAMS["fx"], 0, self.CAM_PARAMS["cx"]],
                           [0, self.CAM_PARAMS["fy"], self.CAM_PARAMS["cy"]],
                           [0, 0, 1]])

    def __len__(self):

        return len(self.df)
    
    def __getitem__(self, index):                      
        
        # load image and mask
        scene = self.scenes[index]
        
        # load image
        rgb = cv2.imread(str(scene["path"] / "rgb" / scene["rgb"]), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # load camera parameters
        scene_camera = scene["scene_camera"]
        scene_camera = json.load(open(scene_camera, "r"))
        scene_camera = scene_camera[f"{index}"]['scene_gt']
        
        # load mask
        mask = cv2.imread(str(scene["path"] / "mask_crushed" / scene["mask"]), cv2.IMREAD_GRAYSCALE)
        masks = [(mask == i).astype(np.uint8) for i in range(1, len(scene_camera) + 1)]

        # extract object from mask
        bboxes = []
        for i in range(len(masks)):

            x, y, w, h = cv2.boundingRect(masks[i])

            x = max(0, x - int(0.1 * w))
            y = max(0, y - int(0.1 * h))
            w = min(rgb.shape[1] - x, int(1.2 * w))
            h = min(rgb.shape[0] - y, int(1.2 * h))

            bboxes.append([x, y, w, h])

        
        # create a mask for the object with 12 points
        mask = [np.zeros((self.N_PTS, 
                          self.CAM_PARAMS["height"], 
                          self.CAM_PARAMS["width"]), 
                          dtype=np.uint8) for _ in range(len(scene_camera))]

        # load mesh
        crops = []
        for i, obj in enumerate(scene_camera):
            
            # load camera parameters
            cam_R_m2c = obj['cam_R_m2c']
            cam_t_m2c = obj['cam_t_m2c']
            obj_id = obj['obj_id']
            
            # load mesh
            mesh = trimesh.load(f'/home/dts/Desktop/PoseEstimation/datasets/icbin_models/obj_00000{obj_id}_target.ply')

            # create transformation matrix
            H = np.eye(4)
            H[:3, :3] = np.array(cam_R_m2c).reshape(3, 3)
            H[:3, 3] = np.array(cam_t_m2c).reshape(3)
            P = np.matmul(self.K, H[:3, :])

            # projects points into image plane
            pts = np.array(mesh.vertices)
            pts = np.hstack([pts, np.ones((pts.shape[0], 1))])
            pts = np.matmul(P, pts.T).T
            pts = pts[:, :2] / pts[:, 2:]
            pts = pts.astype(np.int32)

            for j, pt in enumerate(pts):
                
                if pt[0] >= 0 and pt[0] < self.CAM_PARAMS["width"] and pt[1] >= 0 and pt[1] < self.CAM_PARAMS["height"]:
                    mask[i][j, pt[1], pt[0]] = 1
            
            # dextract the bounding boxes from original mask
            x, y, w, h = bboxes[i]
            mask[i] = mask[i][:, y:y+h, x:x+w]

            # resize all mask[i] to 256x256
            mask[i] = np.array([cv2.resize(m, (256, 256)) for m in mask[i]])

            # put a normal distribution around the points
            mask[i] = np.array([cv2.GaussianBlur(m, (5, 5), 0) for m in mask[i].astype(np.float32)])
            mask[i] = torch.from_numpy(mask[i])

            # superimpose mask[i] on a corresponding crop of the original image
            crop = rgb[y:y+h, x:x+w]
            crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_CUBIC)
            crop = crop.astype(np.float32)
            crop = self.transform(image=crop)["image"]

            # convert crop to tensor
            crop = np.transpose(crop, (2, 0, 1))
            crop = torch.from_numpy(crop)

            crops.append(crop)
        
        # convert crops to tensor
        crops = torch.stack(crops)
        mask = torch.stack(mask)

        return crops, mask

        
if __name__ == "__main__":

    icbin_dataset = ICBINDataset(path="datasets/icbin")
    icbin_dataset[0]

    
