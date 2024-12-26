import os
import json

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from transformers import pipeline
from ultralytics import YOLO

from gen_conversation import get_pretrain_gt_conversations
from number_to_words import number_to_words

from PIL import Image


# Constants
YOLO_MODEL_PATH = "weights/yolov8x-worldv2.pt"
DEPTH_MODEL_PATH = "depth-anything/Depth-Anything-V2-large-hf"


IMAGE_PATH = "/mnt/HDD_1/walker/dlcv_pretrain_raw/nuImages_dataset"
OUTPUT_DEPTH_NPY_PATH = "/mnt/HDD_1/walker/dlcv_pretrain_data/nuImages/npy"
OUTPUT_DEPTH_PNG_PATH = "/mnt/HDD_1/walker/dlcv_pretrain_data/nuImages/png"
OUTPUT_JSON_PATH = "/mnt/HDD_1/walker/dlcv_pretrain_data/nuImages/nuImages_qa.json"
IMAGE_FORMAT = "jpg"

CLASSES = [
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "van",
    "SUV",
    "trailer",
    "moped",
    "ambulance",
    "construction vehicle",
    "pedestrian",
    "cyclist",
    "motorcyclists",
    "road users",
    "red traffic light",
    "traffic light",
    "parking sign",
    "warning traffic sign",
    "directional traffic sign",
    "traffic box",
    "sentry box",
    "traffic cone",
    "traffic island",
    "barrier",
    "bollard",
    "debris",
    "machinery",
    "dustbin",
    "concrete block",
    "dog",
    "chair",
    "phone booth",
    "streetlights"
]

print("CLASS NUM: ", len(CLASSES))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Create results directory
os.makedirs("results", exist_ok=True)


def get_depth_category(depth_value):
    """Get depth category based on depth value."""
    thresholds = {
        1.0: "immediate",
        0.6: "short range",
        0.4: "mid range",
        0.15: "long range",
    }
    for threshold, category in sorted(thresholds.items()):
        if depth_value <= threshold:
            return category


def get_position(bbox, image_width):
    """
    Determine the horizontal position of an object based on its bounding box center
    """
    # Calculate center x-coordinate of the bounding box
    center_x = (bbox[0] + bbox[2]) / 2

    # Define the boundaries for three equal sections
    third_width = image_width / 3

    if center_x < third_width:
        return "left"
    elif center_x < 2 * third_width:
        return "middle"
    else:
        return "right"


final_output = []

# Initialize model and pipeline
detection_model = YOLO(YOLO_MODEL_PATH)
depth_pipe = pipeline("depth-estimation", model=DEPTH_MODEL_PATH, device=DEVICE)

# Set custom classes if defined
if CLASSES:
    detection_model.set_classes(CLASSES)

# Get a list of all file names in the folder
file_names = [f for f in os.listdir(IMAGE_PATH) if f.endswith(f'.{IMAGE_FORMAT}') and os.path.isfile(os.path.join(IMAGE_PATH, f))]


images = []
ids = []
# Execute prediction for specified categories on an image
# for batch_idx, batch in enumerate(tqdm(train_loader)):
BATCH_SIZE = 5

for file_id, file_name in enumerate(file_names):
    image = Image.open(os.path.join(IMAGE_PATH, file_name))
    image_name = file_name.split(".")[0]
    
    if (file_id + 1) % BATCH_SIZE == 0:
        images.append(image)
        ids.append(image_name)
        
        batch = {
            "images": images,
            "ids": ids,
        }
        images = []
        ids = []
    else:
        images.append(image)
        ids.append(image_name)
        continue
        
    
    yolo_results = detection_model.predict(batch["images"], conf=0.15)
    depth_results = depth_pipe(batch["images"])

    for index, (yolo_result, depth_result) in enumerate(
        zip(yolo_results, depth_results)
    ):
        image_name = batch["ids"][index]
        image_data = []
        image_data_for_conversation = []
        
        # initialize category count
        category_count = dict()
        near_category_count = dict()
        for category in CLASSES:
            category_count[category] = 0
            near_category_count[category] = 0
            
        depth_map = np.array(depth_result["predicted_depth"])
        depth_map_tensor = torch.tensor(depth_map)
        
        w, h = batch["images"][index].size
        
        # to resize the depth map back to the original image shape
        depth_map = F.interpolate(
            depth_map_tensor[None, None], (h, w), mode="bilinear", align_corners=False
        )[0, 0]
        depth_map = depth_map.cpu().numpy()
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        depth_map_normalized = (depth_map - depth_min) / (depth_max - depth_min)
        
        # Save as .png file
        # depth_map_png = (depth_map_normalized * 255).astype(np.uint8)
        # cv2.imwrite(f"{OUTPUT_DEPTH_PNG_PATH}/{image_name}.png", depth_map_png)

        # Save as .npy file
        # np.save(f"{OUTPUT_DEPTH_NPY_PATH}/{image_name}.npy", depth_map_normalized)


        for box in yolo_result.boxes:
            score = box.conf.item()
            label = box.cls.item()
            xyxy = box.xyxy[0].tolist()

            # pass to depth detection model
            roi_depth = depth_map[
                int(xyxy[1]) : int(xyxy[3]), int(xyxy[0]) : int(xyxy[2])
            ]
            avg_depth = float(np.mean(roi_depth))
            avg_depth = (avg_depth - depth_min) / (depth_max - depth_min)
            
            object_unique_name = (
                detection_model.names.get(box.cls.item())
                + " "
                + number_to_words(
                    category_count[detection_model.names.get(box.cls.item())] + 1
                )
            )
            category_count[detection_model.names.get(box.cls.item())] += 1
            
            if get_depth_category(avg_depth) == "immediate":
                near_category_count[detection_model.names.get(box.cls.item())] += 1
            
            image_data.append(
                {
                    "category_name": detection_model.names.get(box.cls.item()),
                    "object_unique_name": object_unique_name,
                    "bbox_n": box.xyxyn[0].tolist(),
                    "bbox": xyxy,
                    "confidence": float(score),
                }
            )
            
            image_data_for_conversation.append(
                {
                    "category_name": detection_model.names.get(box.cls.item()),
                    "object_unique_name": object_unique_name,
                    "bbox_n": box.xyxyn[0].tolist(),
                    "bbox": xyxy,
                    "confidence": float(score),
                    "depth_value": avg_depth.item(),
                    "position": get_position(xyxy, w),
                    "depth_category": get_depth_category(avg_depth),
                }
            )
        
        conversations = get_pretrain_gt_conversations(image_name, image_data_for_conversation, category_count, near_category_count)
        if len(conversations) == 0:
            continue
        
        final_output.append(
            {
                "id": image_name,
                "image": f"{IMAGE_PATH}/{image_name}.{IMAGE_FORMAT}",
                "depth_png": f"{OUTPUT_DEPTH_PNG_PATH}/{image_name}.png",
                "depth_npy": f"{OUTPUT_DEPTH_NPY_PATH}/{image_name}.npy",
                "conversations": conversations,
                "bounding_box": image_data
            }
        )
        
        # yolo_result.save(f"results/{image_name}.jpg")


with open(f"{OUTPUT_JSON_PATH}", "w") as f:
    json.dump(final_output, f, indent=4)
    print(f"Saved to {OUTPUT_JSON_PATH}")
    
# with open("pretrain_dataset/nuImages_dataset_qa.json", "w") as f:
#     json.dump(final_output, f, indent=4)
#     print("Saved to pretrain_dataset/nuImages_dataset_qa.json")
