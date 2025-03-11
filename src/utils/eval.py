import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from ultralytics import YOLO
from thop import profile
import torch


def draw_yolo_predictions_vs_ground_truth(
    image,
    boxes,
    masks,
    class_ids,
    true_masks,
    true_class_ids,
    class_names=None,
    colors=None,
):
    """Plots predicted masks on image vs expected masks on image

    Args:
        image (np.ndarray): image
        boxes (Iterable): list of bounding boxes
        masks (Iterable): list of predicted polygones
        class_ids (list):  list of predicted yolo ids
        true_masks (Itrble): Ground Truth
        true_class_ids (Iterable): list of expected yolo class Ids
        class_names (dict, optional): maps id-classname. Defaults to None.
        colors (dict, optional): maps id-color. Defaults to None.
    """
    overlay_pred = image.copy()
    overlay_gt = image.copy()
    height, width = image.shape[:2]
    if colors is None:
        np.random.seed(42)
        colors = {
            i: np.random.randint(0, 255, (3,))
            for i in set(class_ids).union(set(true_class_ids))
        }

    fig, ax = plt.subplots(1, 2, figsize=(18, 6))
    for box, mask, class_id in zip(boxes, masks, class_ids):

        x, y, w, h = box
        x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
        color = colors[class_id]

        # Draw mask
        polygon = Polygon(mask, color=color / 255, alpha=0.4)
        ax[0].add_patch(polygon)

        # Draw bounding box
        cv2.rectangle(overlay_pred, (x1, y1), (x2, y2), color.tolist(), 2)

        # Add label
        label = class_names[class_id] if class_names else f"Class {class_id}"
        cv2.putText(
            overlay_pred,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

    for true_mask, true_class_id in zip(true_masks, true_class_ids):

        # Draw True mask
        true_color = colors[true_class_id]
        poly_array = (true_mask * np.array([width, height])).astype(int)
        true_polygon = Polygon(poly_array, color=true_color / 255, alpha=0.4)
        ax[1].add_patch(true_polygon)

        # Compute bounding box (x_min, y_min, width, height)
        x_min, y_min = np.min(poly_array, axis=0)
        x_max, y_max = np.max(poly_array, axis=0)

        # Draw the bounding box
        cv2.rectangle(
            overlay_gt, (x_min, y_min), (x_max, y_max), true_color.tolist(), 2
        )

        # Add True label
        label = class_names[true_class_id] if class_names else f"Class {true_class_id}"
        cv2.putText(
            overlay_gt,
            label,
            (x_min, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

    ax[0].imshow(overlay_pred)
    ax[1].imshow(overlay_gt)
    ax[0].set_title("Predicted Masks")
    ax[1].set_title("Ground Truth Masks")
    plt.axis("off")
    plt.show()


def measure_inference_time(model_path, test_images, device="cuda"):
    """
    Measures inference time of a YOLO model.

    Args:
        model_path (str): Path to the trained YOLO model.
        test_images (list): List of test image file paths.
        device (str): "cuda" for GPU or "cpu".

    Returns:
        dict: Average inference time and FPS.
    """
    times = []
    model = YOLO(model_path)
    for img in test_images:

        start_time = time.time()
        model.predict(img)  # Run inference
        end_time = time.time()

        times.append(end_time - start_time)

    avg_time = np.mean(times)
    fps = 1 / avg_time

    return {"avg_inference_time (s)": avg_time, "FPS": fps}


def get_model_size(model_path):
    """
    Gets the YOLO model size in MB.

    Args:
        model_path (str): Path to the model file.

    Returns:
        float: Model size in MB.
    """
    size_in_mb = os.path.getsize(model_path) / (1024 * 1024)  # Convert bytes to MB
    return size_in_mb


def compute_flops(model, input_size=(1, 3, 640, 640)):
    """
    Computes FLOPs of a YOLO model.

    Args:
        model_path (str): Path to YOLO model.
        input_size (tuple): Input shape (batch, channels, height, width).

    Returns:
        float: FLOPs in GFLOPs.
    """
    dummy_input = torch.randn(input_size).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    flops, params = profile(model.model, inputs=(dummy_input,))
    return {"FLOPs (GFLOPS)": flops / 1e9, "Params (M)": params / 1e6}


def convert_model(model_path, format, **kwargs):
    """Converts yolo model to the given format

    Args:
        model_path (str): path to model
        format (str): onnx, tflite, coreML...
    """
    model = YOLO(model_path)
    model.export(format=format, **kwargs)
