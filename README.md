# Intel Semiconductor Defect Classification System

## Overview

This project implements an AI-powered defect classification system for
semiconductor images using a deep learning model (EfficientNet-B2) and a
FastAPI-based web application.\
The system supports real-time image inference, visualization of model
performance metrics, and an interactive dashboard interface.

This solution was developed for the **Intel Semiconductor Solutions
Challenge 2026**.

------------------------------------------------------------------------

## Features

-   Real-time image classification using a trained deep learning model
-   FastAPI backend for model inference
-   Interactive web UI for image upload and prediction
-   Visualization of training curves and confusion matrices
-   GPU acceleration support (CUDA)
-   REST API endpoints for integration
-   Modular and production-ready project structure

------------------------------------------------------------------------

## Model Details

Model Architecture: EfficientNet-B2\
Framework: PyTorch\
Input Size: 260 x 260 pixels\
Classes:

-   defect1
-   defect2
-   defect3
-   defect4
-   defect5
-   defect8
-   defect9
-   defect10
-   new_good

Performance Metrics:

-   Test Accuracy: 95.56%
-   Best Validation Accuracy: 97.50%
-   Balanced dataset split: 70% train / 15% validation / 15% test

------------------------------------------------------------------------

## Project Structure

backend/ app/ main.py api/ model.py templates/ index.html assets/
training_curves.png confusion_matrix_val.png confusion_matrix_test.png
models/ best_model.pth

------------------------------------------------------------------------

## Installation

### 1. Clone the repository

git clone `<repository-url>`{=html} cd backend

### 2. Create virtual environment

python -m venv venv

### 3. Activate environment

Windows:

venv`\Scripts`{=tex}`\activate`{=tex}

Mac/Linux:

source venv/bin/activate

### 4. Install dependencies

pip install fastapi uvicorn torch torchvision pillow jinja2
python-multipart

------------------------------------------------------------------------

## Running the Application

Start the FastAPI server:

uvicorn app.main:app --reload

Open the web interface:

http://127.0.0.1:8000

API documentation:

http://127.0.0.1:8000/docs

------------------------------------------------------------------------

## API Endpoints

GET /

Returns dashboard UI

POST /predict

Uploads an image and returns prediction results

GET /health

Checks API health status

------------------------------------------------------------------------

## Example Prediction Response

{ "filename": "sample.png", "predicted_class": "defect8", "confidence":
0.94 }

------------------------------------------------------------------------

## Graphs and Evaluation

The dashboard includes:

-   Training Loss and Accuracy Curves
-   Validation Confusion Matrix
-   Test Confusion Matrix

These plots are automatically generated during model training and stored
in:

app/assets/

------------------------------------------------------------------------

## Hardware Support

-   GPU acceleration (CUDA)
-   CPU fallback if GPU not available
-   Tested on RTX 3060

------------------------------------------------------------------------

## Future Improvements

-   Batch image prediction
-   Model versioning
-   Real-time defect localization
-   Deployment using Docker
-   Cloud inference support

------------------------------------------------------------------------

## License

This project is intended for academic and research purposes.
