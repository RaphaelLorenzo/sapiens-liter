# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import multiprocessing as mp
import os
import time
import warnings
from argparse import ArgumentParser
from collections import defaultdict
from functools import partial
from multiprocessing import cpu_count, Pool, Process
from typing import List, Optional, Sequence, Union

import cv2
import json_tricks as json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.adhoc_image_dataset import AdhocImageDataset
from utils.classes_and_palettes import (
    COCO_KPTS_COLORS,
    COCO_WHOLEBODY_KPTS_COLORS,
    GOLIATH_KPTS_COLORS,
    GOLIATH_SKELETON_INFO,
    COCO_SKELETON_INFO,
    COCO_WHOLEBODY_SKELETON_INFO
)
from utils.pose_utils import nms, top_down_affine_transform, udp_decode

from tqdm import tqdm

from ultralytics import YOLO

from torchvision.transforms.functional import crop
from torchvision.transforms.functional import resize
from PIL import Image


warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")
warnings.filterwarnings("ignore", category=UserWarning, module="json_tricks.encoders")


def crop_resize_and_normalize(orig_img, bboxes_list, input_shape, mean, std, draw_boxes = True, show_bboxes = False):
    """Preprocess pose images and bboxes."""
    preprocessed_images = []
    extended_padded_bboxes = []
    # cropped_images = []
    # Create a new element for each box

    for bbox in bboxes_list:    
            
        # Set the padding ratio
        pad_ratio = 1.25
        desired_aspect_ratio = input_shape[1] / input_shape[0] # width / height of the model

        # Calculate the padded bbox
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        padded_box_width = width * pad_ratio
        padded_box_height = height * pad_ratio
        padded_bbox = np.array([x_center - (padded_box_width / 2), y_center - (padded_box_height / 2), x_center + (padded_box_width / 2), y_center + (padded_box_height / 2)])
        
        # Calculate the extended padded bbox with the desired aspect ratio
        if (padded_box_width / padded_box_height) > desired_aspect_ratio:
            extended_padded_box_height = padded_box_width / desired_aspect_ratio
            extended_padded_box_width = padded_box_width
        else:
            extended_padded_box_width = padded_box_height * desired_aspect_ratio
            extended_padded_box_height = padded_box_height
            
        extended_padded_bbox = np.array([x_center - (extended_padded_box_width / 2), y_center - (extended_padded_box_height / 2), x_center + (extended_padded_box_width / 2), y_center + (extended_padded_box_height / 2)])
        
        # Store the extended padded bbox
        extended_padded_bboxes.append(extended_padded_bbox)
        
        
        # Crop, resize and normalize the image
        orig_img_tensor = torch.from_numpy(orig_img).permute(2, 0, 1) # C, H, W
        cropped_img = crop(orig_img_tensor, int(extended_padded_bbox[1]), int(extended_padded_bbox[0]), int(extended_padded_bbox[3] - extended_padded_bbox[1]), int(extended_padded_bbox[2] - extended_padded_bbox[0])) # shape: (C, H, W)
        cropped_img_resized = resize(cropped_img, (input_shape[0], input_shape[1]), interpolation=Image.BILINEAR) # shape: (C, H, W)
        cropped_img_resized = cropped_img_resized.float()
        mean = torch.Tensor(mean).view(-1, 1, 1)
        std = torch.Tensor(std).view(-1, 1, 1)
        cropped_img_resized = (cropped_img_resized - mean) / std
        
        # Store the preprocessed image
        preprocessed_images.append(cropped_img_resized)
        
        # Draw the bboxes on the image and show it
        if draw_boxes or show_bboxes:
            cv2.rectangle(orig_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
            cv2.putText(orig_img, "original bbox", (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(orig_img, (int(padded_bbox[0]), int(padded_bbox[1])), (int(padded_bbox[2]), int(padded_bbox[3])), (0, 255, 255), 2)
            cv2.putText(orig_img, "padded bbox", (int(padded_bbox[0]), int(padded_bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.rectangle(orig_img, (int(extended_padded_bbox[0]), int(extended_padded_bbox[1])), (int(extended_padded_bbox[2]), int(extended_padded_bbox[3])), (0, 255, 0), 2)
            cv2.putText(orig_img, "padded bbox with desired ratio", (int(extended_padded_bbox[0]), int(extended_padded_bbox[1]+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cropped_img_resized_np = cropped_img_resized.numpy().transpose(1, 2, 0)

            if show_bboxes:
                cv2.imshow("Image with bboxes (type any key to continue)", orig_img)        
                key = cv2.waitKey(0)
                # if key == ord('q') or key == 27:
                #     exit()
                cv2.destroyAllWindows()
                
                cv2.imshow("Cropped, resized and normalized image (type any key to continue)", cropped_img_resized_np)
                key = cv2.waitKey(0)
                # if key == ord('q') or key == 27:
                #     exit()
                cv2.destroyAllWindows()

        
    return preprocessed_images, extended_padded_bboxes


def batch_inference_topdown(
    model: nn.Module,
    imgs: List[Union[np.ndarray, str]],
    flip=False,
    half=False,
):
    with torch.no_grad(), torch.autocast(device_type="cuda"):
        
        if half:
            imgs = imgs.half()
            
        heatmaps = model(imgs.cuda())
        
        if flip:
            heatmaps_ = model(imgs.cuda().flip(-1))
            heatmaps = (heatmaps + heatmaps_) * 0.5
            
        imgs.float().cpu()
        
    return heatmaps.float().cpu()


def save_pose_results(
    img, results, output_path, input_shape, heatmap_scale, kpt_colors, kpt_thr, radius, skeleton_info, thickness
):
    
        
    # pred_instances_list = split_instances(result)
    heatmap = results["heatmaps"]
    extended_padded_bboxes = results["extended_padded_bboxes"]
    has_dets = results["has_dets"]
    img_shape = img.shape
    instance_keypoints = []
    instance_scores = []

    for i in range(len(heatmap)):
        result = udp_decode(
            heatmap[i].cpu().unsqueeze(0).float().data[0].numpy(),
            input_shape,
            (int(input_shape[0] / heatmap_scale), int(input_shape[1] / heatmap_scale)),
        )

        keypoints, keypoint_scores = result
        
        # remap keypoints to their image coordinates
        image_bbox = extended_padded_bboxes[i] # xmin, ymin, xmax, ymax in original image coordinates
        image_bbox_width = image_bbox[2] - image_bbox[0]
        image_bbox_height = image_bbox[3] - image_bbox[1]
        bbox_size = np.array([image_bbox_width, image_bbox_height])
        
        keypoints = (keypoints / input_shape) * bbox_size + image_bbox[:2]
        
        instance_keypoints.append(keypoints[0])
        
        if has_dets:
            instance_scores.append(keypoint_scores[0])
        else:
            instance_scores.append(keypoint_scores[0]*0.0) # set the scores to 0 for the samples that have no detections

    pred_save_path = output_path.replace(".jpg", ".json").replace(".png", ".json")

    with open(pred_save_path, "w") as f:
        json.dump(
            dict(
                instance_info=[
                    {
                        "keypoints": keypoints.tolist(),
                        "keypoint_scores": keypoint_scores.tolist(),
                    }
                    for keypoints, keypoint_scores in zip(
                        instance_keypoints, instance_scores
                    )
                ]
            ),
            f,
            indent="\t",
        )
    # img = pyvips.Image.new_from_array(img)
    instance_keypoints = np.array(instance_keypoints).astype(np.float32)
    instance_scores = np.array(instance_scores).astype(np.float32)

    keypoints_visible = np.ones(instance_keypoints.shape[:-1])
    for kpts, score, visible in zip(
        instance_keypoints, instance_scores, keypoints_visible
    ):
        kpts = np.array(kpts, copy=False)

        if (
            kpt_colors is None
            or isinstance(kpt_colors, str)
            or len(kpt_colors) != len(kpts)
        ):
            raise ValueError(
                f"the length of kpt_color "
                f"({len(kpt_colors)}) does not matches "
                f"that of keypoints ({len(kpts)})"
            )

        # draw each point on image
        for kid, kpt in enumerate(kpts):
            if score[kid] < kpt_thr or not visible[kid] or kpt_colors[kid] is None:
                # skip the point that should not be drawn
                continue

            color = kpt_colors[kid]
            if not isinstance(color, str):
                color = tuple(int(c) for c in color[::-1])
            img = cv2.circle(img, (int(kpt[0]), int(kpt[1])), int(radius), color, -1)
        
        # draw skeleton
        for skid, link_info in skeleton_info.items():
            pt1_idx, pt2_idx = link_info['link']
            color = link_info['color'][::-1] # BGR

            pt1 = kpts[pt1_idx]; pt1_score = score[pt1_idx]
            pt2 = kpts[pt2_idx]; pt2_score = score[pt2_idx]

            if pt1_score > kpt_thr and pt2_score > kpt_thr:
                x1_coord = int(pt1[0]); y1_coord = int(pt1[1])
                x2_coord = int(pt2[0]); y2_coord = int(pt2[1])
                cv2.line(img, (x1_coord, y1_coord), (x2_coord, y2_coord), color, thickness=thickness)

    cv2.imwrite(output_path, img)

def fake_pad_images_to_batchsize(imgs, batch_size):
    return F.pad(imgs, (0, 0, 0, 0, 0, 0, 0, batch_size - imgs.shape[0]), value=0)

def get_sample_bboxes(result, resize_ratio_width, resize_ratio_height, bbox_thr):
    sample_bboxes = []
    classes = result.boxes.cls.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()  # tensor of shape (n, 4)
    for i in range(len(boxes)):
        if classes[i] == 0 and confs[i] > bbox_thr:
            xmin = boxes[i][0]
            xmax = boxes[i][2]
            ymin = boxes[i][1]
            ymax = boxes[i][3]
            
            xmin_scaled = xmin * resize_ratio_width
            ymin_scaled = ymin * resize_ratio_height    
            xmax_scaled = xmax * resize_ratio_width
            ymax_scaled = ymax * resize_ratio_height
            
            sample_bboxes.append(np.array([xmin_scaled, ymin_scaled, xmax_scaled, ymax_scaled]))
    
    return sample_bboxes

def main(args, model_path):
    
    input_shape = (3, 1024, 768) # C, H, W
    
    start = time.time()

    ### Create the output directory ###
    if not os.path.exists(args.output):
        os.makedirs(args.output)



    ### Build the model from a checkpoint file ###
    pose_estimator = torch.jit.load(model_path)
    if args.half:
        pose_estimator = pose_estimator.half()
        
    pose_estimator = pose_estimator.to(args.device)
    
    detection_model = YOLO("yolo11x.pt")


    ### Get the input images from the path ###
    input = args.input
    image_names = []

    if os.path.isdir(input):
        input_dir = input  # Set input_dir to the directory specified in input
        image_names = [
            image_name
            for image_name in sorted(os.listdir(input_dir))
            if image_name.endswith(".jpg") or image_name.endswith(".png")
        ]
    elif os.path.isfile(input) and input.endswith(".txt"):
        # If the input is a text file, read the paths from it and set input_dir to the directory of the first image
        with open(input, "r") as file:
            image_paths = [line.strip() for line in file if line.strip()]
        image_names = [
            os.path.basename(path) for path in image_paths
        ]  # Extract base names for image processing
        input_dir = (
            os.path.dirname(image_paths[0]) if image_paths else ""
        )  # Use the directory of the first image path
    elif os.path.isfile(input) and (input.endswith(".mp4") or input.endswith(".avi")):
        raise NotImplementedError("Video input not implemented")
    else:
        raise ValueError("Invalid input type {}".format(input))


    
    ### Declare the dataset and dataloader. Do not add preprocessing like normalization because it is done later ###
    inference_dataset = AdhocImageDataset(
        [os.path.join(input_dir, img_name) for img_name in image_names],
    ) 
    
    inference_dataloader = torch.utils.data.DataLoader(
        inference_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(min(args.batch_size, cpu_count()) // 4, 4),
    )


    ### Define the keypoints colors and skeleton info ###
    if args.num_keypoints == 17:
        KPTS_COLORS = COCO_KPTS_COLORS
        SKELETON_INFO = COCO_SKELETON_INFO
    elif args.num_keypoints == 308:
        KPTS_COLORS = GOLIATH_KPTS_COLORS
        SKELETON_INFO = GOLIATH_SKELETON_INFO
    elif args.num_keypoints == 133:
        KPTS_COLORS = COCO_WHOLEBODY_KPTS_COLORS
        SKELETON_INFO = COCO_WHOLEBODY_SKELETON_INFO
    else:
        raise ValueError("Invalid number of keypoints {}".format(args.num_keypoints))


    
    ### Run the inference ###
    for batch_idx, (batch_image_name, batch_orig_imgs, batch_imgs) in tqdm(
        enumerate(inference_dataloader), total=len(inference_dataloader)
    ):
        
        orig_img_shape = batch_orig_imgs.shape
        valid_images_len = len(batch_orig_imgs)
        
        bboxes_batch = []
        
        ### Resize images to 640x640 for YOLO detection ###
        batch_imgs_resize = F.interpolate(batch_imgs, size=(640, 640), mode='bilinear', align_corners=False)
        resize_ratio_width =  orig_img_shape[2] / 640
        resize_ratio_height =  orig_img_shape[1] / 640
        batch_imgs_resize = batch_imgs_resize.float() / 255.0
        
        detection_results = detection_model(batch_imgs_resize, verbose=False)

        ### Get the bounding boxes from the detection model for each sample ###
        for sample_idx, result in enumerate(detection_results):
            sample_bboxes = get_sample_bboxes(result, resize_ratio_width, resize_ratio_height, args.bbox_thr)
            bboxes_batch.append(sample_bboxes)  
        
        if np.all([len(bboxes) == 0 for bboxes in bboxes_batch]):
            # No detections in this batch
            continue
        
        ### Store the number of detections in each sample and add fake empty detections for the samples that have no detections ###
        img_bbox_map = {}
        has_dets = []
        for i, bboxes in enumerate(bboxes_batch):
            if len(bboxes) == 0:
                img_bbox_map[i] = 1 # one for the fake bbox
                has_dets.append(False)
                bboxes_batch[i] = np.array(
                    [[0, 0, orig_img_shape[1], orig_img_shape[2]]] # orig_img_shape: B H W C
                )
            else:
                img_bbox_map[i] = len(bboxes)
                has_dets.append(True)
                
        ### Preprocess the images for the inference ###
        pose_imgs, pose_imgs_extended_padded_bboxes = [], []
        for k in range(len(bboxes_batch)):
            preprocessed_images, extended_padded_bboxes = crop_resize_and_normalize(batch_orig_imgs[k].numpy(), bboxes_batch[k], (input_shape[1], input_shape[2]), [123.5, 116.5, 103.5], [58.5, 57.0, 57.5])
                        
            pose_imgs.extend(preprocessed_images)
            pose_imgs_extended_padded_bboxes.extend(extended_padded_bboxes)


        ### Compute the number of batches to run for the inference, because of the multiple detections in each sample ###
        n_pose_batches = (len(pose_imgs) + args.batch_size - 1) // args.batch_size


        ### Run the inference ###
        torch.compiler.cudagraph_mark_step_begin() # use this to tell torch compiler the start of model invocation as in 'flip' mode the tensor output is overwritten
        pose_results = []
        for i in range(n_pose_batches):
            imgs = torch.stack(
                pose_imgs[i * args.batch_size : (i + 1) * args.batch_size], dim=0
            )
            
            valid_len = len(imgs)
            imgs = fake_pad_images_to_batchsize(imgs, args.batch_size)
            
            heatmaps = batch_inference_topdown(pose_estimator, imgs, flip=args.flip, half=args.half)[:valid_len]
            pose_results.extend(heatmaps)
                    
        ### Store the results in a dictionary for the valid ones ###
        batched_results = []
        for idx, (_, bbox_len) in enumerate(img_bbox_map.items()):
            result = {
                "heatmaps": pose_results[:bbox_len].copy(),
                "extended_padded_bboxes": pose_imgs_extended_padded_bboxes[:bbox_len].copy(),
                "has_dets": has_dets[idx]
            }
            batched_results.append(result)
            del (
                pose_results[:bbox_len],
                pose_imgs_extended_padded_bboxes[:bbox_len],
            )

        assert len(batched_results) == len(batch_orig_imgs)


        ### Post-process and save the results ###
        for i, r, img_name in zip(
            batch_orig_imgs[:valid_images_len],
            batched_results[:valid_images_len],
            batch_image_name,
        ):
            save_pose_results(i.numpy(), 
                             r, 
                             os.path.join(args.output, os.path.basename(img_name)), 
                             (input_shape[2], input_shape[1]), 
                             4, # heatmap scale
                             KPTS_COLORS, 
                             args.kpt_thr, 
                             args.radius,  
                             SKELETON_INFO, 
                             args.thickness)

    total_time = time.time() - start
    fps = 1 / ((time.time() - start) / len(image_names))
    print(
        f"\033[92mTotal inference time: {total_time:.2f} seconds. FPS: {fps:.2f}\033[0m"
    )



MODEL_KPTS_SIZE_TO_NAME = {
    "17": {
        "03b": "sapiens_0.3b_coco_best_coco_AP_796_torchscript.pt2",
        "06b": "sapiens_0.6b_coco_best_coco_AP_812_torchscript.pt2",
        "1b": "sapiens_1b_coco_best_coco_AP_821_torchscript.pt2",
    },
    "133": {
        "03b": "sapiens_0.3b_coco_wholebody_best_coco_wholebody_AP_620_torchscript.pt2",
        "06b": "sapiens_0.6b_coco_wholebody_best_coco_wholebody_AP_695_torchscript.pt2",
        "1b": "sapiens_1b_coco_wholebody_best_coco_wholebody_AP_727_torchscript.pt2",
    },
    "308": {
        "03b": "sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2",
        "06b": "sapiens_0.6b_goliath_best_goliath_AP_609_torchscript.pt2",
        "1b": "sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2",
    },
}


if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument(
        "--model", "-m", help="Type of SAPIENS pose model to use among, 03b, 06b, 1b", default="03b"
    )
    parser.add_argument(
        "--input", "-i", type=str, default="./examples/input", help="Image/Image directory/Video file"
    )
    parser.add_argument(
        "--num_keypoints", "-nk", type=int, default=133, help="Number of keypoints in the pose model. Used for visualization"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="./examples/output_pose", help="output dir for visualization images and other files"
    )
    parser.add_argument(
        "--batch_size", "-b", type=int, default=2, help="Set batch size to do batch inference. Default 48"
    )
    parser.add_argument(
        "--half", "-hp", action="store_true", default=False, help="use half precision for inference"
    )
    parser.add_argument(
        "--device", "-d", default="cuda:0", help="Device used for inference. Default cuda:0"
    )
    parser.add_argument(
        "--bbox_thr", type=float, default=0.3, help="Bounding box score threshold. Default 0.3"
    )
    parser.add_argument(
        "--kpt_thr", type=float, default=0.3, help="Visualizing keypoint thresholds"
    )
    parser.add_argument(
        "--radius", type=int, default=2, help="Keypoint radius for visualization"
    )
    parser.add_argument(
        "--thickness", type=int, default=1, help="Keypoint skeleton thickness for visualization"
    )
    parser.add_argument(
        "--heatmap_scale", type=int, default=4, help="Heatmap scale for keypoints. Image to heatmap ratio" # TODO : check what it is ?
    )
    parser.add_argument(
        "--flip", "-f", action="store_true", default=False, help="Flip the input image horizontally and inference again"
    )

    args = parser.parse_args()
    
    assert(args.model in ["03b", "06b", "1b", "2b"]), "Invalid model, expected 03b, 06b, 1b"
    assert(args.num_keypoints in [17, 133, 308]), "Invalid number of keypoints, expected 17, 133, 308"
    
    model_name = MODEL_KPTS_SIZE_TO_NAME[str(args.num_keypoints)][args.model]
    model_path = os.path.join(os.environ["SAPIENS_LITE_CHECKPOINT_ROOT"], "torchscript", "pose", "checkpoints", model_name)
    print("Using model : {} at {} ({} kpt, size {})".format(model_name, model_path, args.num_keypoints, args.model))
    
    assert(os.path.exists(model_path)), "Model path does not exist ({})".format(model_path)
    assert(os.path.exists(args.input)), "Input path does not exist ({})".format(args.input)
    assert args.output != "", "Output path is empty"

    main(args, model_path)
