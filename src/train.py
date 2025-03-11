import argparse
from ultralytics import YOLO


def train(model_name="yolov8n-seg.pt", data_yaml="./data.yaml", device="0", **kwargs):
    """Train yolo model. (Or resume training in case of interruption)

    Args:
        model_name (str, optional): name of yolo model or path to it. Defaults to "yolov8n-seg.pt".
        data_yaml (str, optional): path to yolo data config file. Defaults to "./data.yaml".
        device (str, optional): "cpu" or "0"for GPU. Defaults to "0".
    """
    model = YOLO(model_name)
    results = model.train(data=data_yaml,
                          device=device,
                          **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Passes yolo train arguments")

    # Define expected arguments
    parser.add_argument("--model_name", type=str, help="Yolo path or model name")
    parser.add_argument("--data_yaml", type=str, help="Path to data.yaml file")
    parser.add_argument("--device", type=str, help="Device where to train the model")

    # Parse known arguments
    args, unknown = parser.parse_known_args()

    # Convert known arguments to a dictionary
    args_dict = vars(args)

    # Process unknown arguments (convert to dict)
    unknown_dict = {}
    for i in range(0, len(unknown), 2):  
        if i + 1 < len(unknown) and unknown[i].startswith("--"):
            key = unknown[i].lstrip("--")
            value = unknown[i + 1]
            unknown_dict[key] = value

    # Merge both dictionaries
    final_kwargs = {**args_dict, **unknown_dict}
    
    train(**final_kwargs)