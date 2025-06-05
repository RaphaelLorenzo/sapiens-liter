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
from adhoc_image_dataset import AdhocImageDataset
from classes_and_palettes import (
    COCO_KPTS_COLORS,
    COCO_WHOLEBODY_KPTS_COLORS,
    GOLIATH_KPTS_COLORS,
    GOLIATH_SKELETON_INFO,
    COCO_SKELETON_INFO,
    COCO_WHOLEBODY_SKELETON_INFO
)
from pose_utils import nms, top_down_affine_transform, udp_decode

from tqdm import tqdm

from worker_pool import WorkerPool

# try:
#     from mmdet.apis import inference_detector, init_detector
#     from mmdet.structures import DetDataSample, SampleList
#     from mmdet.utils import get_test_pipeline_cfg

#     has_mmdet = True
# except (ImportError, ModuleNotFoundError):
#     has_mmdet = False

from ultralytics import YOLO



warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning, module="mmengine")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")
warnings.filterwarnings("ignore", category=UserWarning, module="json_tricks.encoders")

timings = {}
BATCH_SIZE = 1


def preprocess_pose(orig_img, bboxes_list, input_shape, mean, std):
    """Preprocess pose images and bboxes."""
    preprocessed_images = []
    centers = []
    scales = []
    for bbox in bboxes_list:
        img, center, scale = top_down_affine_transform(orig_img.copy(), bbox)
        img = cv2.resize(
            img, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_LINEAR
        ).transpose(2, 0, 1)
        img = torch.from_numpy(img)
        img = img[[2, 1, 0], ...].float()
        mean = torch.Tensor(mean).view(-1, 1, 1)
        std = torch.Tensor(std).view(-1, 1, 1)
        img = (img - mean) / std
        preprocessed_images.append(img)
        centers.extend(center)
        scales.extend(scale)
    return preprocessed_images, centers, scales


def batch_inference_topdown(
    model: nn.Module,
    imgs: List[Union[np.ndarray, str]],
    dtype=torch.bfloat16,
    flip=False,
):
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        heatmaps = model(imgs.cuda())
        if flip:
            heatmaps_ = model(imgs.to(dtype).cuda().flip(-1))
            heatmaps = (heatmaps + heatmaps_) * 0.5
        imgs.cpu()
    return heatmaps.cpu()


def img_save_and_vis(
    img, results, output_path, input_shape, heatmap_scale, kpt_colors, kpt_thr, radius, skeleton_info, thickness
):
    # pred_instances_list = split_instances(result)
    heatmap = results["heatmaps"]
    centres = results["centres"]
    scales = results["scales"]
    img_shape = img.shape
    instance_keypoints = []
    instance_scores = []
    # print(scales[0], centres[0])
    for i in range(len(heatmap)):
        result = udp_decode(
            heatmap[i].cpu().unsqueeze(0).float().data[0].numpy(),
            input_shape,
            (int(input_shape[0] / heatmap_scale), int(input_shape[1] / heatmap_scale)),
        )

        keypoints, keypoint_scores = result
        keypoints = (keypoints / input_shape) * scales[i] + centres[i] - 0.5 * scales[i]
        instance_keypoints.append(keypoints[0])
        instance_scores.append(keypoint_scores[0])

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

def fake_pad_images_to_batchsize(imgs):
    return F.pad(imgs, (0, 0, 0, 0, 0, 0, 0, BATCH_SIZE - imgs.shape[0]), value=0)

def load_model(checkpoint, use_torchscript=False):
    if use_torchscript:
        return torch.jit.load(checkpoint)
    else:
        return torch.export.load(checkpoint).module()

def main():
    """Visualize the demo images.
    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument("pose_checkpoint", help="Checkpoint file for pose")
    parser.add_argument("--det-config", default="", help="Config file for detection")
    parser.add_argument("--det-checkpoint", default="", help="Checkpoint file for detection")
    parser.add_argument("--input", type=str, default="", help="Image/Video file")
    parser.add_argument(
        "--num_keypoints",
        type=int,
        default=133,
        help="Number of keypoints in the pose model. Used for visualization",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        default=[1024, 768],
        help="input image size (height, width)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="",
        help="root of the output img file. "
        "Default not saving the visualization images.",
    )
    parser.add_argument(
        "--batch_size",
        "--batch-size",
        type=int,
        default=48,
        help="Set batch size to do batch inference. ",
    )
    parser.add_argument(
        "--fp16", action="store_true", default=False, help="Model inference dtype"
    )
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--det-cat-id",
        type=int,
        default=0,
        help="Category id for bounding box detection model",
    )
    parser.add_argument(
        "--bbox-thr", type=float, default=0.3, help="Bounding box score threshold"
    )
    parser.add_argument(
        "--nms-thr", type=float, default=0.3, help="IoU threshold for bounding box NMS"
    )
    parser.add_argument(
        "--kpt-thr", type=float, default=0.3, help="Visualizing keypoint thresholds"
    )
    parser.add_argument(
        "--radius", type=int, default=9, help="Keypoint radius for visualization"
    )
    parser.add_argument(
        "--thickness", type=int, default=-1, help="Keypoint skeleton thickness for visualization"
    )
    parser.add_argument(
        "--heatmap-scale", type=int, default=4, help="Heatmap scale for keypoints. Image to heatmap ratio"
    )
    parser.add_argument(
        "--flip",
        type=bool,
        default=False,
        help="Flip the input image horizontally and inference again",
    )

    args = parser.parse_args()

    # Replace the detector with YOLO
    # if args.det_config is None or args.det_config == "":
    #     use_det = False
    # else:
    #     use_det = True
    #     assert has_mmdet, "Please install mmdet to run the demo."
    #     assert args.det_checkpoint is not None

    #     from detector_utils import (
    #         adapt_mmdet_pipeline,
    #         init_detector,
    #         process_images_detector,
    #     )
    
    if args.det_config == "yolo":
        use_det = True
        print("Use YOLO detector from ultralytics")
        detection_model = YOLO("yolo11x.pt")
    elif args.det_config == "yolo-seg":
        use_det = True
        print("Use YOLO-Seg detector from ultralytics")
        raise NotImplementedError("YOLO-Seg detector not implemented")
    else:
        use_det = False
        print("Do not use detector")

    assert args.input != ""
    ## if skeleton thickness is not specified, use radius as thickness
    if args.thickness == -1:
        args.thickness = args.radius

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3,) + tuple(args.shape)
    else:
        raise ValueError("invalid input shape")

    print("Using input shape", input_shape)

    mp.log_to_stderr()
    torch._inductor.config.force_fuse_int_mm_with_mul = True
    torch._inductor.config.use_mixed_mm = True

    start = time.time()

    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root)

    assert args.output_root != ""
    args.pred_save_path = (
        f"{args.output_root}/results_"
        f"{os.path.splitext(os.path.basename(args.input))[0]}.json"
    )

    # build detector
    # if use_det:
    #     detector = init_detector(
    #         args.det_config, args.det_checkpoint, device=args.device
    #     )
    #     detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # build pose estimator
    USE_TORCHSCRIPT = '_torchscript' in args.pose_checkpoint

    # build the model from a checkpoint file
    pose_estimator = load_model(args.pose_checkpoint, USE_TORCHSCRIPT)

    ## no precision conversion needed for torchscript. run at fp32
    if not USE_TORCHSCRIPT:
        dtype = torch.half if args.fp16 else torch.bfloat16
        pose_estimator.to(dtype)
        pose_estimator = torch.compile(pose_estimator, mode="max-autotune", fullgraph=True)
    else:
        dtype = torch.float32  # TorchScript models use float32
        pose_estimator = pose_estimator.to(args.device)

    global BATCH_SIZE
    BATCH_SIZE = args.batch_size

    input = args.input
    image_names = []

    # Check if the input is a directory or a text file
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

    scale = args.heatmap_scale
    inference_dataset = AdhocImageDataset(
        [os.path.join(input_dir, img_name) for img_name in image_names],
    )  # do not provide preprocess args for detector as we use mmdet
    inference_dataloader = torch.utils.data.DataLoader(
        inference_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(min(args.batch_size, cpu_count()) // 4, 4),
    )
    
    # Replace the multiprocessing with a single process
    # pose_preprocess_pool = WorkerPool(
    #     preprocess_pose, processes=max(min(args.batch_size, cpu_count()), 1)
    # )
    img_save_pool = WorkerPool(
        img_save_and_vis, processes=max(min(args.batch_size, cpu_count()), 1)
    )

    KPTS_COLORS = COCO_WHOLEBODY_KPTS_COLORS  ## 133 keypoints
    SKELETON_INFO = COCO_WHOLEBODY_SKELETON_INFO

    if args.num_keypoints == 17:
        KPTS_COLORS = COCO_KPTS_COLORS
        SKELETON_INFO = COCO_SKELETON_INFO
    elif args.num_keypoints == 308:
        KPTS_COLORS = GOLIATH_KPTS_COLORS
        SKELETON_INFO = GOLIATH_SKELETON_INFO

    for batch_idx, (batch_image_name, batch_orig_imgs, batch_imgs) in tqdm(
        enumerate(inference_dataloader), total=len(inference_dataloader)
    ):

        print(batch_idx) # int
        print(batch_image_name) # tuple of strings
        print("batch_orig_imgs", batch_orig_imgs.shape, type(batch_orig_imgs)) # tensor of size [B, H, W, 3]
        print("batch_imgs", batch_imgs.shape, type(batch_imgs)) # tensor of size [B, 3, H, W]
        
        # batch_orig_imgs = batch_orig_imgs[:,:640,:640,:]
        # batch_imgs = batch_imgs[:,:,:640,:640]
        
        orig_img_shape = batch_orig_imgs.shape
        valid_images_len = len(batch_orig_imgs)
        if use_det:
            # imgs = batch_orig_imgs.clone()[
            #     ..., [2, 1, 0]
            # ]  # since detector uses mmlab, directly use original images
            # bboxes_batch = process_images_detector(args, imgs.numpy(), detector)
            # get bbox in format [x1, y1, x2, y2]
            
            bboxes_batch = []
            
            # Resize images to 640x640 for YOLO detection
            batch_imgs_resize = F.interpolate(batch_imgs, size=(640, 640), mode='bilinear', align_corners=False)
            resize_ratio_width =  orig_img_shape[2] / 640
            resize_ratio_height =  orig_img_shape[1] / 640
            
            detection_results = detection_model(batch_imgs_resize)

            # boxes_per_sample_idx = {} # list of length B each element is a numpy array of shape (N,6)
        
            for sample_idx, result in enumerate(detection_results):
                
                sample_bboxes = []
                
                # person_boxes = []
                classes = result.boxes.cls.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()  # tensor of shape (n, 4)
                for i in range(len(boxes)):
                    if classes[i] == 0 and confs[i] > 0.5:
                        xmin = boxes[i][0]
                        xmax = boxes[i][2]
                        ymin = boxes[i][1]
                        ymax = boxes[i][3]
                        
                        xmin_scaled = xmin * resize_ratio_width
                        ymin_scaled = ymin * resize_ratio_height    
                        xmax_scaled = xmax * resize_ratio_width
                        ymax_scaled = ymax * resize_ratio_height
                        
                        sample_bboxes.append(np.array([xmin_scaled, ymin_scaled, xmax_scaled, ymax_scaled]))
                        
                bboxes_batch.append(sample_bboxes)
                # for i in range(len(boxes)):
                #     if classes[i] == 0 and confs[i] > 0.5:
                #         # add conf and class to the box
                #         detection = np.concatenate([boxes[i], np.array([confs[i], classes[i]])])
                #         person_boxes.append(detection)
            
            if len(bboxes_batch) == 0 or len(bboxes_batch[0]) == 0:
                continue
            else:
                print("bboxes_batch", bboxes_batch[0])
            # boxes_per_sample_idx[sample_idx] = np.asarray(person_boxes)
            
            if True:
                for sample_idx in range(len(bboxes_batch)):
                    
                    image_npy = batch_orig_imgs.cpu().numpy()[sample_idx,...]
                    for bbox in bboxes_batch[sample_idx]:
                        cv2.rectangle(image_npy, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)

                    cv2.imshow("image", image_npy)

                    key = cv2.waitKey(0)
                    if key == ord('q') or key == 27:
                        exit()
                
        else:
            bboxes_batch = [[] for _ in range(len(batch_orig_imgs))]
        
        assert len(bboxes_batch) == valid_images_len

        for i, bboxes in enumerate(bboxes_batch):
            if len(bboxes) == 0:
                bboxes_batch[i] = np.array(
                    [[0, 0, orig_img_shape[1], orig_img_shape[2]]] # orig_img_shape: B H W C
                )

        img_bbox_map = {}
        for i, bboxes in enumerate(bboxes_batch):
            img_bbox_map[i] = len(bboxes)

        # Replace the multiprocessing with a single process
        # args_list = [
        #     (
        #         i,
        #         bbox_list,
        #         (input_shape[1], input_shape[2]),
        #         [123.5, 116.5, 103.5],
        #         [58.5, 57.0, 57.5],
        #     )
        #     for i, bbox_list in zip(batch_orig_imgs.numpy(), bboxes_batch)
        # ]

        # print("args_list", len(args_list))
        # pose_ops = pose_preprocess_pool.run(args_list)
        # pose_ops = preprocess_pose(batch_orig_imgs.numpy, bboxes_batch, input_shape, [123.5, 116.5, 103.5], [58.5, 57.0, 57.5])
        # print("pose_ops", len(pose_ops))

        pose_imgs, pose_img_centers, pose_img_scales = [], [], []
        for k in range(batch_orig_imgs.shape[0]):
            op = preprocess_pose(batch_orig_imgs[k].numpy(), bboxes_batch[k], (input_shape[1], input_shape[2]), [123.5, 116.5, 103.5], [58.5, 57.0, 57.5])
                        
            pose_imgs.extend(op[0])
            pose_img_centers.extend(op[1])
            pose_img_scales.extend(op[2])

        n_pose_batches = (len(pose_imgs) + args.batch_size - 1) // args.batch_size

        # use this to tell torch compiler the start of model invocation as in 'flip' mode the tensor output is overwritten
        torch.compiler.cudagraph_mark_step_begin()  
        pose_results = []
        for i in range(n_pose_batches):
            imgs = torch.stack(
                pose_imgs[i * args.batch_size : (i + 1) * args.batch_size], dim=0
            )
            
            if False:
                print("imgs for inference", imgs.shape)
                np_image = imgs.cpu().numpy().transpose(0, 2, 3, 1)
                print("np_image", np_image.shape, np_image.mean(), np_image.std(), np_image.max(), np_image.min(), np_image.dtype)

                cv2.imshow("np_image", np_image[0])
                key = cv2.waitKey(0)
                if key == ord('q') or key == 27:
                    exit()
            
            valid_len = len(imgs)
            imgs = fake_pad_images_to_batchsize(imgs)
            pose_results.extend(
                batch_inference_topdown(pose_estimator, imgs, dtype=dtype)[:valid_len]
            )
            
        batched_results = []
        for _, bbox_len in img_bbox_map.items():
            result = {
                "heatmaps": pose_results[:bbox_len].copy(),
                "centres": pose_img_centers[:bbox_len].copy(),
                "scales": pose_img_scales[:bbox_len].copy(),
            }
            batched_results.append(result)
            del (
                pose_results[:bbox_len],
                pose_img_centers[:bbox_len],
                pose_img_scales[:bbox_len],
            )

        assert len(batched_results) == len(batch_orig_imgs)

        args_list = [
            (
                i.numpy(),
                r,
                os.path.join(args.output_root, os.path.basename(img_name)),
                (input_shape[2], input_shape[1]),
                scale,
                KPTS_COLORS,
                args.kpt_thr,
                args.radius,
                SKELETON_INFO,
                args.thickness,
            )
            for i, r, img_name in zip(
                batch_orig_imgs[:valid_images_len],
                batched_results[:valid_images_len],
                batch_image_name,
            )
        ]
        img_save_pool.run_async(args_list)

    # pose_preprocess_pool.finish()
    img_save_pool.finish()

    total_time = time.time() - start
    fps = 1 / ((time.time() - start) / len(image_names))
    print(
        f"\033[92mTotal inference time: {total_time:.2f} seconds. FPS: {fps:.2f}\033[0m"
    )


if __name__ == "__main__":
    main()
