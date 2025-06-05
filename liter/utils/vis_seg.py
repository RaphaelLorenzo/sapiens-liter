# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gc
import multiprocessing as mp
import os
import time
from argparse import ArgumentParser
from functools import partial
from multiprocessing import cpu_count, Pool, Process
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from adhoc_image_dataset import AdhocImageDataset
from classes_and_palettes import GOLIATH_CLASSES, GOLIATH_PALETTE
from tqdm import tqdm

from worker_pool import WorkerPool

torchvision.disable_beta_transforms_warning()

from ultralytics import YOLO


timings = {}
BATCH_SIZE = 32


def warmup_model(model, batch_size):
    imgs = torch.randn(batch_size, 3, 1024, 768).to(dtype=model.dtype).cuda()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=model.dtype
    ):
        for i in range(3):
            model(imgs)
    torch.cuda.current_stream().wait_stream(s)
    imgs = imgs.detach().cpu().float().numpy()
    del imgs, s

def inference_model(model, imgs, dtype=torch.bfloat16):
    with torch.no_grad():
        if True:
            results = model(imgs.to(dtype).cuda().half())
        else:
            results = model(imgs.to(dtype).cuda())
        imgs.cpu()

    results = [r.cpu() for r in results]

    return results


def fake_pad_images_to_batchsize(imgs):
    return F.pad(imgs, (0, 0, 0, 0, 0, 0, 0, BATCH_SIZE - imgs.shape[0]), value=0)


def img_save_and_viz(
    image, result, output_path, classes, palette, title=None, opacity=0.5, threshold=0.3, 
):
    
    print("SAVING !")
    
    output_file = (
        output_path.replace(".jpg", ".png")
        .replace(".jpeg", ".png")
        .replace(".png", ".npy")
    )
    output_seg_file = (
        output_path.replace(".jpg", ".png")
        .replace(".jpeg", ".png")
        .replace(".png", "_seg.npy")
    )

    image = image.data.numpy() ## bgr image

    seg_logits = F.interpolate(
        result.unsqueeze(0), size=image.shape[:2], mode="bilinear"
    ).squeeze(0)

    if seg_logits.shape[0] > 1:
        pred_sem_seg = seg_logits.argmax(dim=0, keepdim=True)
    else:
        seg_logits = seg_logits.sigmoid()
        pred_sem_seg = (seg_logits > threshold).to(seg_logits)

    pred_sem_seg = pred_sem_seg.data[0].numpy()

    mask = pred_sem_seg > 0
    np.save(output_file, mask)
    np.save(output_seg_file, pred_sem_seg)

    num_classes = len(classes)
    sem_seg = pred_sem_seg
    ids = np.unique(sem_seg)[::-1]
    legal_indices = ids < num_classes
    ids = ids[legal_indices]
    labels = np.array(ids, dtype=np.int64)

    colors = [palette[label] for label in labels]

    mask = np.zeros_like(image)
    for label, color in zip(labels, colors):
        mask[sem_seg == label, :] = color
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    vis_image = (image_rgb * (1 - opacity) + mask * opacity).astype(np.uint8)

    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    vis_image = np.concatenate([image, vis_image], axis=1)
    cv2.imwrite(output_path, vis_image)
    print("Writing image to ", output_path)

def load_model(checkpoint, use_torchscript=False):
    if use_torchscript:
        return torch.jit.load(checkpoint)
    else:
        return torch.export.load(checkpoint).module()

def main():
    parser = ArgumentParser()
    parser.add_argument("checkpoint", help="Checkpoint file")
    parser.add_argument("--input", help="Input image dir")
    parser.add_argument(
        "--output_root", "--output-root", default=None, help="Path to output dir"
    )
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--batch_size",
        "--batch-size",
        type=int,
        default=4,
        help="Set batch size to do batch inference. ",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        default=[1024, 768],
        help="input image size (height, width)",
    )
    parser.add_argument(
        "--fp16", action="store_true", default=False, help="Model inference dtype"
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.5,
        help="Opacity of painted segmentation map. In (0, 1] range.",
    )
    parser.add_argument("--title", default="result", help="The image identifier.")
    args = parser.parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3,) + tuple(args.shape)
    else:
        raise ValueError("invalid input shape")

    mp.log_to_stderr()
    torch._inductor.config.force_fuse_int_mm_with_mul = True
    torch._inductor.config.use_mixed_mm = True

    start = time.time()

    USE_TORCHSCRIPT = '_torchscript' in args.checkpoint

    # build the model from a checkpoint file
    exp_model = load_model(args.checkpoint, USE_TORCHSCRIPT)
    
    if True:
        exp_model = exp_model.half()
    
    ## no precision conversion needed for torchscript. run at fp32
    if not USE_TORCHSCRIPT:
        dtype = torch.half if args.fp16 else torch.bfloat16
        exp_model.to(dtype)
        exp_model = torch.compile(exp_model, mode="max-autotune", fullgraph=True)
    else:
        dtype = torch.float32  # TorchScript models use float32
        exp_model = exp_model.to(args.device)

    input = args.input
    image_names = []

    # Check if the input is a directory or a text file
    if os.path.isdir(input):
        input_dir = input  # Set input_dir to the directory specified in input
        image_names = [
            image_name
            for image_name in sorted(os.listdir(input_dir))
            if image_name.endswith(".jpg")
            or image_name.endswith(".png")
            or image_name.endswith(".jpeg")
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
    else:
        raise ValueError("Invalid input, must be a directory or a text file")

    if len(image_names) == 0:
        raise ValueError("No images found in the input directory")

    # If left unspecified, create an output folder relative to this script.
    if args.output_root is None:
        args.output_root = os.path.join(input_dir, "output")

    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root)

    global BATCH_SIZE
    BATCH_SIZE = args.batch_size

    n_batches = (len(image_names) + args.batch_size - 1) // args.batch_size

    inference_dataset = AdhocImageDataset(
        [os.path.join(input_dir, img_name) for img_name in image_names],
        # (input_shape[1], input_shape[2]),
        # mean=[123.5, 116.5, 103.5],
        # std=[58.5, 57.0, 57.5],
    )
    inference_dataloader = torch.utils.data.DataLoader(
        inference_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(min(args.batch_size, cpu_count()), 1),
    )
    total_results = []
    image_paths = []
    img_save_pool = WorkerPool(
        img_save_and_viz, processes=max(min(args.batch_size, cpu_count()), 1)
    )
    
    detection_model = YOLO("yolo11x.pt")
    
    for batch_idx, (batch_image_name, batch_orig_imgs, batch_imgs) in tqdm(
        enumerate(inference_dataloader), total=len(inference_dataloader)
    ):
        
        # if batch_idx < 150:
        #     continue

        print("batch_orig_imgs", batch_orig_imgs.shape, type(batch_orig_imgs), batch_orig_imgs.dtype) # tensor of size [B, H, W, 3]
        print("batch_imgs", batch_imgs.shape, type(batch_imgs), batch_imgs.dtype) # tensor of size [B, 3, H, W]
        
        orig_img_shape = batch_orig_imgs.shape
        
        batch_imgs_resize = F.interpolate(batch_imgs, size=(640, 640), mode='bilinear', align_corners=False)
        resize_ratio_width =  orig_img_shape[2] / 640
        resize_ratio_height =  orig_img_shape[1] / 640
        
        bboxes_batch = []
        
        detection_results = detection_model(batch_imgs_resize)

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
        
        # for sample_idx in range(len(bboxes_batch)):
        #     img = batch_orig_imgs[sample_idx].cpu().numpy() # shape [H, W, 3]
            
        #     for bbox in bboxes_batch[sample_idx]:
        #         cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
            
        #     cv2.imshow("img", img)
        #     key = cv2.waitKey(0)
        #     if key == ord('q') or key == 27:
        #         exit()
                    
        
        if len(bboxes_batch) == 0 or len(bboxes_batch[0]) == 0:
            print("no bboxes found", bboxes_batch)
            continue
        
        # for each element of the batch take only the first bbox and crop the image
        batch_imgs_cropped_with_ar = []
        batch_orig_imgs_cropped_with_ar = []
        
        batch_imgs_resized = []
        batch_orig_imgs_resized = []
        
        
        for sample_idx in range(len(bboxes_batch)):
            bbox = bboxes_batch[sample_idx][0]
            
            desired_aspect_ratio = 1024 / 768
            padding_ratio = 1.25
            bbox_h = bbox[3] - bbox[1]
            bbox_w = bbox[2] - bbox[0]
            bbox_h_padded = bbox_w * padding_ratio
            bbox_w_padded = bbox_h * padding_ratio
            
            bbox_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            
            if bbox_h_padded > bbox_w_padded * desired_aspect_ratio:
                bbox_w_padded = bbox_h_padded / desired_aspect_ratio
            else:
                bbox_h_padded = bbox_w_padded * desired_aspect_ratio
                
            bbox_xmin = int(bbox_center[0] - bbox_w_padded / 2)
            bbox_ymin = int(bbox_center[1] - bbox_h_padded / 2)
            bbox_xmax = int(bbox_center[0] + bbox_w_padded / 2)
            bbox_ymax = int(bbox_center[1] + bbox_h_padded / 2)
            
            
            # Get image dimensions
            h, w = batch_orig_imgs[sample_idx].shape[:2]
            
            # Clip bbox coordinates to image bounds
            bbox_ymin_clipped = max(0, int(bbox_ymin))
            bbox_ymax_clipped = min(h, int(bbox_ymax))
            bbox_xmin_clipped = max(0, int(bbox_xmin))
            bbox_xmax_clipped = min(w, int(bbox_xmax))
            
            # Create padded output images
            pad_h = bbox_ymax - bbox_ymin
            pad_w = bbox_xmax - bbox_xmin
            
            print("pad_h", pad_h)
            print("pad_w", pad_w)
            
            orig_img_cropped_with_ar = torch.zeros((pad_h, pad_w, 3), dtype=torch.uint8)
            img_cropped_with_ar = torch.zeros((3, pad_h, pad_w), dtype=torch.float32)
            
            # Calculate padding offsets
            y_start = max(0, -int(bbox_ymin))
            x_start = max(0, -int(bbox_xmin))
            
            print("x_start", x_start)
            print("y_start", y_start)
            
            print("x size :", x_start + (bbox_xmax_clipped - bbox_xmin_clipped) - x_start)
            print("y size :", y_start + (bbox_ymax_clipped - bbox_ymin_clipped) - y_start)

            print("x size target :", bbox_xmax_clipped - bbox_xmin_clipped)
            print("y size target :", bbox_ymax_clipped - bbox_ymin_clipped)
            
            print("bbox_ymin_clipped", bbox_ymin_clipped, "bbox_ymax_clipped", bbox_ymax_clipped, "vs image size", batch_imgs.shape[2])
            print("bbox_xmin_clipped", bbox_xmin_clipped, "bbox_xmax_clipped", bbox_xmax_clipped, "vs image size", batch_imgs.shape[3])
            test = batch_imgs[sample_idx, :, bbox_ymin_clipped:bbox_ymax_clipped,
                                        bbox_xmin_clipped:bbox_xmax_clipped]
            
            print("test", test.shape)
            
            
            # Copy valid region
            orig_img_cropped_with_ar[y_start:y_start + (bbox_ymax_clipped - bbox_ymin_clipped), 
                           x_start:x_start + (bbox_xmax_clipped - bbox_xmin_clipped)] = \
                batch_orig_imgs[sample_idx, bbox_ymin_clipped:bbox_ymax_clipped, 
                                           bbox_xmin_clipped:bbox_xmax_clipped]
                                           
            img_cropped_with_ar[:, y_start:y_start + (bbox_ymax_clipped - bbox_ymin_clipped),
                          x_start:x_start + (bbox_xmax_clipped - bbox_xmin_clipped)] = \
                batch_imgs[sample_idx, :, bbox_ymin_clipped:bbox_ymax_clipped,
                                        bbox_xmin_clipped:bbox_xmax_clipped]
                
                
            batch_orig_imgs_cropped_with_ar.append(orig_img_cropped_with_ar)
            batch_imgs_cropped_with_ar.append(img_cropped_with_ar)
            
            # resize to 1024x768
            orig_img_cropped_with_ar = F.interpolate(orig_img_cropped_with_ar.permute(2, 0, 1).unsqueeze(0), size=(1024, 768), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)
            img_cropped_with_ar = F.interpolate(img_cropped_with_ar.unsqueeze(0), size=(1024, 768), mode='bilinear', align_corners=False).squeeze(0)
            
            batch_orig_imgs_resized.append(orig_img_cropped_with_ar)
            batch_imgs_resized.append(img_cropped_with_ar)
        

        batch_orig_imgs_cropped_with_ar = torch.stack(batch_orig_imgs_cropped_with_ar)
        batch_imgs_cropped_with_ar = torch.stack(batch_imgs_cropped_with_ar)
        
        print("batch_orig_imgs_cropped_with_ar", batch_orig_imgs_cropped_with_ar.shape, type(batch_orig_imgs_cropped_with_ar))
        print("batch_imgs_cropped_with_ar", batch_imgs_cropped_with_ar.shape, type(batch_imgs_cropped_with_ar))

        batch_orig_imgs_resized = torch.stack(batch_orig_imgs_resized) # shape [B, 1024, 768, 3]
        batch_imgs_resized = torch.stack(batch_imgs_resized) # shape [B, 3, 1024, 768]
        
        print("batch_orig_imgs_resized", batch_orig_imgs_resized.shape, type(batch_orig_imgs_resized))
        print("batch_imgs_resized", batch_imgs_resized.shape, type(batch_imgs_resized))
        
        mean=[123.5, 116.5, 103.5]
        std=[58.5, 57.0, 57.5]
        
        mean = torch.tensor(mean).view(1, 3, 1, 1)  # Shape [1, 3, 1, 1] to match [B, C, H, W]
        std = torch.tensor(std).view(1, 3, 1, 1)    # Shape [1, 3, 1, 1] to match [B, C, H, W]
        batch_imgs_resized = (batch_imgs_resized - mean) / std
        

        valid_images_len = len(batch_imgs_resized)
        batch_imgs_resized = fake_pad_images_to_batchsize(batch_imgs_resized)
        result = inference_model(exp_model, batch_imgs_resized, dtype=dtype)
        
        
        result = [r.float() for r in result]
        
        print("result", result[0].shape, result[0].dtype)


        for i,r,img_name in zip(batch_orig_imgs_resized[:valid_images_len], result[:valid_images_len], batch_image_name):
            img_save_and_viz(i, r, os.path.join(args.output_root, os.path.basename(img_name)), GOLIATH_CLASSES, GOLIATH_PALETTE, args.title, args.opacity)

        # wtf pool does not work when half precision is used
        # args_list = [
        #     (
        #         i,
        #         r,
        #         os.path.join(args.output_root, os.path.basename(img_name)),
        #         GOLIATH_CLASSES,
        #         GOLIATH_PALETTE,
        #         args.title,
        #         args.opacity,
        #     )
        #     for i, r, img_name in zip(
        #         batch_orig_imgs_resized[:valid_images_len],
        #         result[:valid_images_len],
        #         batch_image_name,
        #     )
        # ]
        # img_save_pool.run_async(args_list)

    img_save_pool.finish()

    total_time = time.time() - start
    fps = 1 / ((time.time() - start) / len(image_names))
    print(
        f"\033[92mTotal inference time: {total_time:.2f} seconds. FPS: {fps:.2f}\033[0m"
    )


if __name__ == "__main__":
    main()
