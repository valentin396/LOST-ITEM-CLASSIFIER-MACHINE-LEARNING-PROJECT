# Lost Item Classifier

An image classifier that recognizes common lost-and-found items — built as a prototype feature for [MIZERO](https://github.com/valentin396/MIZERO-AI-FULL-STACK-PROJECT), an AI-powered lost-and-found platform for Rwandan communities.

## What it does

Given a photo of a lost item, the model predicts which category it belongs to (bottle, umbrella, calculator, charger, or key) with a confidence score. The idea: when someone reports a lost item with a photo, this auto-tags its category instead of requiring manual selection — feeding directly into MIZERO's matching logic.

## How it works

- **Transfer learning** on MobileNetV2, pretrained on ImageNet, with a custom classification head fine-tuned on labeled lost-item photos
- Trained on 3,500+ images across 5 categories, sourced from the [Lost and Found dataset on Roboflow Universe](https://universe.roboflow.com/kst-lo6da/lost-and-found-nlb4x) (CC BY 4.0)
- Includes a conversion script (`convert_yolo_to_classification.py`) that turns YOLO-format bounding-box annotations into cropped, classification-ready images

## Results

- **~98% validation accuracy**
- Train and validation loss converge closely with no overfitting

## Tech stack

Python · TensorFlow / Keras · MobileNetV2 · Jupyter Notebook

## Files

- `lost_item_classifier.ipynb` — full training pipeline: data loading, model building, training, evaluation, and inference
- `convert_yolo_to_classification.py` — converts YOLO object-detection format into folder-per-class classification data

## Next steps

- Retrain on real user-submitted photos from MIZERO as they come in, to improve real-world accuracy
- Expand categories to match actual usage patterns (phone, wallet, ID card, etc.)
- Wrap the trained model in an API endpoint for MIZERO's backend
