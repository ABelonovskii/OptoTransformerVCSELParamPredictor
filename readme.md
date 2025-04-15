# OptoTransformer for VCSEL Parameters Prediction

<div style="text-align:center;">
    <img src="vcsel-transformer-logo.png" width="50%" alt="Program Screenshot">
</div>

## Project Overview
This project applies a transformer-based machine learning model to predict various parameters of Vertical-Cavity Surface-Emitting Lasers (VCSELs) from their structural data. The implementation leverages deep learning techniques to understand and predict critical performance metrics from VCSEL architectures. For a detailed theoretical background, you can refer to our article here:
> [A. Belonovskii, E. Girshova, E. Lähderanta, M. Kaliteevski, Predicting VCSEL Emission Properties using Transformer Neural Networks. Laser Photonics Rev 2025, 2401636.](https://onlinelibrary.wiley.com/doi/abs/10.1002/lpor.202401636)

## Demo
To experience the capabilities of the OptoTransformer firsthand, visit my interactive web interface available at:
[OptoTransformer Demo](https://loading-opto-transformer.onrender.com)

This demo allows you to use a pre-trained transformer model to predict VCSEL parameters directly through your web browser. It's a great way to quickly see the practical applications of our research and the effectiveness of the transformer model in real-time prediction.


## Features
- **Data Processing**: Convert raw VCSEL, DBR, or Fabri-Perot data into a format suitable for machine learning models.
- **Model Training**: Train the transformer model on processed VCSEL data.
- **Evaluation and Testing**: Assess model performance on a split test dataset.
- **Prediction**: Predict VCSEL parameters from new data samples.

## Installation
1. Clone this repository.
2. Install required Python packages:
```bash
pip install -r requirements.txt
```

## Usage

The OptoTransformer project is designed for flexibility, allowing users to operate through both command line arguments and configuration files. This dual approach ensures that users can quickly switch between different operational modes and experimental setups without altering the code.

To run the project, you can use the following commands:
- **Training the Model**:
```bash
python main.py --train
```
- **Testing the Model**:
```bash
python main.py --test
```
- **Making Predictions**:
```bash
python main.py --predict
```

Each command loads configurations from `config.yaml`, but can be overridden by additional command line arguments, providing a dynamic and user-friendly interface for various experimental needs.

## Configuration
Modify `config.yaml` as per your requirements.
Modify `model_params.yaml` as per your requirements for model.

### Copyright

© 2024, Aleksei Belonovskii
