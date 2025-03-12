import os
import cv2
import json
import numpy as np


def resize_image(image, r_aspect_ratio=3):
    """Resizes an image based on a given aspect ratio

    Args:
        image (np.ndarray): image to be resized
        r_aspect_ratio (int, optional): reverse target aspect ratio. Defaults to 3.

    Returns:
        np.ndarray: resized image
    """
    new_width = image.shape[1] // r_aspect_ratio
    new_height = image.shape[0] // r_aspect_ratio

    resized_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )

    return resized_image


def resize_polygon(polygon, r_aspect_ratio=3):
    """Resizes a polygon

    Args:
        polygon (Iterator): list of point boundaries of the polygon
        r_aspect_ratio (int, optional): reverse target aspect ratio. Defaults to 3.

    Returns:
        Iterator: list of new computed coordintes
    """

    resized_polygon = [(x // r_aspect_ratio, y // r_aspect_ratio) for x, y in polygon]

    return resized_polygon


def read_annotation(ann_path):
    """Reads instance annotations from the given path

    Args:
        ann_path (str): path to the json file annotation

    Returns:
        dictionary: content of the annotation
    """
    assert ".json" in ann_path, f"Expected the path to be a json file"

    with open(ann_path, "r") as f:
        annotation = json.load(f)

    return annotation


def read_image(image_path):
    """Reads a given image

    Args:
        image_path (str): path to the image

    Returns:
        np.ndarray: RGB image.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def save_image(image: np.ndarray, image_path: str) -> bool:
    """Writes an image in RGB format to the given path.

    Returns:
        bool: True if the image was written successfully else False.
    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.imwrite(image_path, image)


def create_data_yaml(root_dir, classes, train_path, val_path, dataset_path):
    """
    Create the data.yaml file expected by yolo. It has the format shown below
    Args:
        root_dir(str): root of the project
        classes(dict): mapping of id-classname
        train_path(str): path to training images
        val_path(str): path to validation images


    path: ../datasets/coco8-seg # dataset root dir
    train: images/train # train images (relative to 'path')
    val: images/val # val images (relative to 'path')

    names:
        0: person
        1: bicycle
        2: car
        # ...
    """

    with open(os.path.join(dataset_path, "data.yaml"), "w", encoding="UTF-8") as f:
        f.write(f"path: {root_dir} \ntrain: {train_path}\nval: {val_path}\n\nnames:")
        f.writelines(
            ["\n    " + str(id) + ": " + name for (id, name) in classes.items()]
        )


def convert_annotation_to_yolo(
    src, dest, origId_yoloId_dict, resize=True, r_aspect_ratio=3
):
    """converts annotations from MVTec D2S Dataset format
    to YOLO format


    Args:
        src (str): src folder of the MVTec D2S Dataset annotations
        dest (str): destination folder
        origId_yoloId_dict (_type_): mapping from the original classIds to ordered indexes (expected by yolo)
        resize (bool, optional): Whether to resize the annotations. Defaults to True.
        r_aspect_ratio (int, optional): Reverse Aspect ratio by what to resize the annotations. Defaults to 3.
    """
    for root, subdirs, filenames in os.walk(src):

        for filename in filenames:
            abs_filename = os.path.join(root, filename)
            # annotation: {description, tags, size:{height, width},
            #        objects: [{classId,.., points: {exterior: [...], interior: []}}]}
            ann = read_annotation(abs_filename)
            h, w = ann["size"]["height"], ann["size"]["width"]

            polygones = list(
                map(lambda instance: instance["points"]["exterior"], ann["objects"])
            )  # [[[x,y],[x,y],[x,y]],]
            classes = list(map(lambda instance: instance["classId"], ann["objects"]))

            if resize:
                h, w = h // r_aspect_ratio, w // r_aspect_ratio
                polygones = [
                    resize_polygon(polygon, r_aspect_ratio=r_aspect_ratio)
                    for polygon in polygones
                ]

            # normalize polygones [(x1,y1),(x2,y2),...(xn,yn)]
            polygones = [
                np.round(np.array(polygon) / np.array([w, h]), decimals=4)
                for polygon in polygones
            ]

            # validate the normalization
            for polygon in polygones:
                assert np.all(
                    polygon < 1
                ), f"Polygones have not been normalized correctly. Some values > 1"

            # convert to yolo format: class_id x1 y1 x2 y2 x3 y3 .... xn yn
            lines = []

            for class_id, polygon in zip(classes, polygones):

                yolo_id = origId_yoloId_dict[class_id]
                polygon = polygon.astype(str)

                line = (
                    str(yolo_id)
                    + " "
                    + " ".join([" ".join(point) for point in polygon])
                    + "\n"
                )
                lines.append(line)

            # print(f'Lines: {lines}')

            # Save the annotation to .txt file
            # filenames are the same as the images except for the additional .json
            dest_filename = os.path.join(dest, filename.split(".")[0] + ".txt")
            with open(dest_filename, "w") as f:
                f.writelines(lines)


def resize_images_folder(src, dest, r_aspect_ratio=3):
    """Resize a folder of images and saves it to dest.

    Args:
        src (str): source folder
        dest (str): destination folder
        r_aspect_ratio (int, optional): reverse aspect ratio. Defaults to 3.
    """
    for root, _, filenames in os.walk(src):

        for filename in filenames:
            try:
                abs_filename = os.path.join(root, filename)
                image = read_image(abs_filename)

                resized_image = resize_image(image, r_aspect_ratio=r_aspect_ratio)
                saved = save_image(resized_image, os.path.join(dest, filename))
                if not saved:
                    print(f"Failed to save image: {abs_filename}")
                else:
                    print(f"Saved filename: {os.path.join(dest, filename)}")
            except Exception as e:
                print(f"Error : Failed to resize and save image: {filename}:\n {e}")


def read_yolo_annotation(ann_path):
    """Reads annotations of yolo dataset format

    Args:
        ann_path (str): path to the target annotation

    Returns:
        list: list of dictionaries containing the classId and the corresponding polygon
    """
    with open(ann_path, "r") as f:
        lines = f.readlines()

    instances = []

    for idx, line in enumerate(lines):

        class_id, *polygon = line.split(" ")
        instances.append(
            {
                "classId": class_id,
                "polygon": np.array(polygon).astype(float).reshape(-1, 2),
            }
        )  # save as numpy to facilitate restoring the coordinates later

    return instances
