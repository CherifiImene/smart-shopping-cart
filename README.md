# 🛒 Smart Shopping Cart

Smart Shopping Cart is a deep learning project that fine-tunes YOLO on a sample set from the MVTec D2S Dataset to perform instance segmentation on market products. This repository provides Jupyter notebooks for data exploration, preprocessing, training, and evaluation, along with scripts for training and inference.

## 📌 Features

- Fine-tuned YOLO model for instance segmentation of market products.

- **Notebooks** for:

    - Data exploration & analysis 📊

    - Data preprocessing 🛠️

    - Model training 🎯

    - Model evaluation 📈

- **Scripts** for:

    - Training the model 🏋️

    - Running inference on a single image or a folder of images 🖼️

- Support for GPU acceleration (e.g., Google Colab, local GPU setups).

## 📂 Repository Structure

````
📦 smart-shopping-cart
├── notebooks/               # Jupyter notebooks for different stages of the pipeline
│   ├── data_exploration.ipynb
│   ├── data_preprocessing.ipynb
│   ├── train.ipynb          # Notebook for training yolov8n-seg
│   ├── train_yolo11n.ipynb  # Notebook for training yolo11-seg 
│    ├── evaluation.ipynb
├── scripts/                 # Python scripts for automation
│   ├── train.py             # Train YOLO model
│   ├── predict.py           # Run inference on images
├── yolo/                
│    ├── datasets/           # MVTec D2S dataset sample in yolo format
│    │    ├── images/
│    │    ├── labels/
│    │    ├── data.yaml
│    ├── runs/
│        ├── segments/       # History of training yolo models 
                             # and saved best checkpoints

├── README.md                # Project documentation
├── requirements.txt         # Required dependencies
````
## 🔧 Installation

1️⃣ Clone the repository

````
git clone https://github.com/yourusername/smart-shopping-cart.git
cd smart-shopping-cart
````

2️⃣ Install dependencies

It's recommended to use a virtual environment:

````python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

pip install -r requirements.txt
````

## 🏋️ Training the Model

To train YOLO on the dataset, run:

````
python scripts/train.py --epochs 200 --img-size 640
````

## 🔍 Running Inference

Single Image Prediction

````
python scripts/predict.py --image_path path/to/image.jpg
````

Batch Prediction (Folder of Images)

````
python scripts/predict.py --image_path path/to/images/
````

## 📊 Model Evaluation

After training, use the evaluation notebook to assess performance:

Box & Segmentation Precision/Recall

mAP50 and mAP50-95 scores

Confusion matrix & loss curves

## 📦 Dataset

We use a subset of the MVTec D2S Dataset, a dataset for instance segmentation in retail environments.

Go to  **notebooks/data_exploration.ipynb** for instructions on how to install the dataset 

## 🚀 Future Improvements

🔄 Support for additional market product datasets

📱 Deploy the model for real-time mobile applications

⏳ Optimize inference speed on edge devices

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## 📜 License

This project is licensed under the MIT License.

💡 If you find this project useful, give it a ⭐ on GitHub!