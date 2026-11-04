# Multimodal AI for Personalized Cancer Diagnosis

Skin lesion classification on HAM10000 using dermoscopy images and patient metadata
(age, sex, lesion localization), with Grad-CAM explanations and a Streamlit demo.

## Method
- **Image model:** ResNet-18 (ImageNet pretrained), 7 classes
- **Clinical model:** MLP on age, sex and localization (19 encoded features, 32-d embedding)
- **Fusion model:** 512-d image features + 32-d clinical embedding, concatenated (544-d) and classified
- **Explainability:** Grad-CAM on ResNet-18 `layer4`
- **Split:** by `lesion_id` (no leakage), shared across all experiments (`data/split.csv`)
- **Imbalance:** class-weighted cross-entropy

## Results (test set, 1502 images)
| Model | Accuracy | Macro-F1 | Melanoma recall |
|---|---|---|---|
| Image only | 0.779 | 0.658 | 0.557 |
| Clinical only | 0.365 | 0.219 | 0.150 |
| Fusion | 0.812 | 0.718 | 0.647 |

Grad-CAM examples: `results/gradcam_examples.png`, `results/gradcam_errors.png`

## Run the demo
    pip install -r requirements.txt
    streamlit run app.py

Models are loaded from `models/` (`resnet18_best.pth`, `fusion_best.pth`, `clinical_preprocessor.joblib`).

## Repo structure
- `app.py`, `fusion_utils.py`: Streamlit app, model classes, Grad-CAM
- `models/`: trained weights and preprocessor
- `results/`: metrics, predictions, confusion matrices, plots
- `notebooks/`: Kaggle notebooks (training and evaluation)
- `data/split.csv`: train/val/test split

## Data
HAM10000 (Kaggle: "Skin Cancer MNIST: HAM10000"). Images are not included in this repo.

## Team
Member 1: image classification | Member 2: clinical model | Member 3: fusion and evaluation | Member 4: Grad-CAM, app, integration

## Disclaimer
Research prototype for academic use. Not a medical device and not for clinical decisions.