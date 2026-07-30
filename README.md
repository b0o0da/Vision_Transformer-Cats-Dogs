# ViT Baseline — Cat vs Dog Classification

A **Vision Transformer (ViT)** built from scratch with TensorFlow/Keras — patch embeddings, a learnable class token, positional encoding, and a stack of Transformer encoder blocks — trained to classify images as **cat** or **dog**. Served through an interactive **Streamlit** app.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Overview

This project implements a Vision Transformer **from scratch** (no pretrained backbone) to explore how the original ViT architecture performs as a baseline on a simple binary image classification task:

1. Images are split into fixed-size **16×16 patches**.
2. Each patch is linearly projected into an embedding, a learnable **[CLS] token** is prepended, and learnable **positional embeddings** are added.
3. The sequence passes through **4 Transformer encoder blocks** (multi-head self-attention + MLP, each with residual connections and layer normalization).
4. The final [CLS] token representation is passed through a small MLP head to predict cat vs. dog.

## 🗂️ Dataset

- **Source**: [Cat and Dog Images for Classification](https://www.kaggle.com/datasets/ashfakyeafi/cat-dog-images-for-classification) (Kaggle)
- Images resized to `224×224`, normalized to `[0, 1]`.
- Split into train / validation / test (stratified, 64% / 16% / 20%).

## 🧠 Model Architecture

| Hyperparameter     | Value |
|---------------------|-------|
| Image size          | 224×224 |
| Patch size          | 16×16 (→ 196 patches) |
| Embedding dimension | 128 |
| Attention heads      | 4 |
| Transformer blocks  | 4 |
| MLP hidden dim       | 256 |
| Dropout              | 0.2 |
| Output               | Sigmoid (binary: cat/dog) |

Training uses `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint` callbacks, saving only the best model.

## 📊 Evaluation

The notebook reports:
- Training/validation accuracy and loss curves.
- A confusion matrix and full classification report on the test set.

## 🚀 Streamlit App

Upload a cat or dog photo and the app will:
- Split it into patches the same way as training.
- Run it through the ViT model.
- Show the predicted class, confidence, and a probability bar chart.

### Run locally

```bash
git clone https://github.com/<your-username>/vit-baseline-cat-dog.git
cd vit-baseline-cat-dog
pip install -r requirements.txt
```

Place your trained model file inside a `models/` folder:

```
models/
└── best_vit.weights.h5
```

> This model is small (~8 MB), so it's committed directly to the repo instead of being hosted externally — no download step needed.

Then launch the app:

```bash
streamlit run app.py
```

## 📁 Project Structure

```
.
├── vit-baseline-how-it-goes.ipynb   # Training & evaluation notebook
├── vit_model.py                      # ViT architecture (patch embedding, transformer blocks)
├── app.py                            # Streamlit inference app
├── requirements.txt                  # Python dependencies
├── models/
│   └── best_vit.weights.h5           # Trained model (~8 MB)
└── README.md
```

## 🛠️ Tech Stack

- TensorFlow / Keras (custom ViT implementation, no external ViT library)
- Streamlit (deployment)

## 📄 License

This project is released under the MIT License.

---

## 🇸🇦 نبذة بالعربي

المشروع ده بيبني **Vision Transformer (ViT)** من الصفر (من غير أي backbone جاهز) باستخدام TensorFlow/Keras، عشان يصنّف الصور بين **قطط وكلاب**. الصورة بتتقسّم لـ patches صغيرة، وكل patch بتتحول لـ embedding، وبيتضاف ليها CLS token و positional encoding، وبعدين بتعدي على 4 طبقات Transformer encoder. في تطبيق Streamlit بترفع فيه صورة والموديل يقولك قطة ولا كلب مع نسبة الثقة.
