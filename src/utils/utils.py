import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from utils.data import read_image, read_yolo_annotation, save_image


def plot_image_instance_mask(image, polygon, figsize=(8, 4)):
    """
    Draws a polygon on top of an image

    Args:
        image (_type_): _description_
        polygon (_type_): _description_
        figsize (tuple, optional): _description_. Defaults to (8,4).

    """

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    polygon = Polygon([*polygon], alpha=0.9)
    ax.add_patch(polygon)

    ax.imshow(image, alpha=0.8)
    plt.show()

def draw_bbox(image,boxes, class_ids, colors, class_names=None):
    
    for box, class_id in zip(boxes, class_ids):

        x, y, w, h = box
        x1, y1, x2, y2 = int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)
        color = colors[class_id]

        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color.tolist(), 2)

        # Add label
        label = class_names[class_id] if class_names else f"Class {class_id}"
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    return image

def plot_image_polygones(image_path, polygones, colors):
    """Plots polygones on an image using opencv and saves the results

    Args:
        image_path (str): path to image
        polygones (list): list points in boundary
    """
    
    image = read_image(image_path=image_path)
    
    for polygon , color in zip(polygones, colors):
        
        if len(polygon.shape) < 3:
            polygon = polygon.reshape((-1,1,2))
        
        isClosed = True
        thickness = 2
        image = cv2.polylines(image, [polygon], 
                      isClosed, color, thickness)
    
    dest_path = image_path.split(".")+"_poly.png"
    saved = save_image(image=image,image_path=dest_path)
    print(f'Status image with overladed polygons saved to path: {dest_path}: {saved}')
                       
def plot_yolo_images_labels(image_paths, label_paths, 
                            subplot=(2, 5), 
                            title="Training Resized samples"):
    """Plots a list of images with their corresponding annotations in yolo format.

    Args:
        image_paths (list): list of image paths
        label_paths (list): list of label paths
        subplot (tuple, optional): Number of subplots needed. Depends on the number of images. 
                                    Defaults to (2, 5).
        title (str, optional): Title of the graph. Defaults to "Training Resized samples".
    """
    fig, axes = plt.subplots(*subplot, figsize=(16, 8))
    fig.suptitle(title)
    i, j = 0, 0
    for idx, (image_p, ann_p) in enumerate(zip(image_paths, label_paths)):

        image = read_image(image_p)
        instances = read_yolo_annotation(ann_p)
        h, w = image.shape[:2]
        i, j = idx // 5, idx % 5
        axes[i, j].imshow(image, alpha=0.8)
        for instance in instances:

            # print(h_color)
            points = (instance["polygon"] * np.array([w, h])).astype(int)
            # print(f"points: {points}")
            polygon = Polygon([*points], alpha=0.9)

            axes[i, j].add_patch(polygon)

    plt.show()
