import argparse
import os
from ultralytics import YOLO
import cv2
import torch
import numpy as np
from utils.data import read_image
from utils.utils import draw_bbox, plot_image_polygones

def predict(model, image_path):
    """Performs instance segmentation on an image

    Args:
        model (YOLO): instance segmentation model
        image_path (str): path to image
    """
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')
    
    model = model.to(device)  
    
    
    image = read_image(image_path)  
    results = model(image)  # predict on an image
    # Access the results
    xy = results[0].masks.xy  # mask in polygon format
    boxes = results[0].boxes.xywh.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    class_names = [model.names[class_id] for class_id in class_ids]
    colors = {i: np.random.randint(0, 255, (3,)) for i in set(class_ids)}
    
    image = draw_bbox(image, boxes,class_ids, colors, class_names)
    plot_image_polygones(image, xy, colors)
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Pass multiple keyword arguments for prediction")

    parser.add_argument("--model_path", required=False, default="./yolo/runs/segment/train11/weights/best.pt", type=str, help="Path to the model")
    parser.add_argument("--image_path", required=True,type=int, help="Path to image")

    # Parsing arguments and converting to dictionary
    args = vars(parser.parse_args())  # Converts argparse.Namespace to dictionary
    
    if not os.path.exists(args["image_path"]):
        
        raise FileNotFoundError(f'Path: {args["image_path"]} not found!')
    
    if not os.path.exists(args["model_path"]):
        
        raise FileNotFoundError(f'Path: {args["model_path"]} not found!')
    
    model = YOLO(args['model_path'])
    
    image_path = args['image_path']
    if os.path.isdir(image_path):
        filenames = list(filter(lambda filename: filename.split(".")[-1] in [".png",".PNG",".jpeg", ".jpg"] , os.listdir(image_path)))
        
        filenames = list(map(lambda f: os.path.join(image_path,f), filenames))
        
        for f in filenames:
            
            predict(model, f)
    else:
        predict(model, image_path)
            
    