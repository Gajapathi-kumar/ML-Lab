# SmartVision ML - Image Classification System

A one-page web application (Streamlit) that covers the full machine learning pipeline:
Load Image -> Preprocessing -> Feature Extraction -> Training -> Performance Evaluation (+ Predict).

Algorithms: SVM, Random Forest, KNN. Features: HOG, intensity histogram, raw pixels.

## How to run (no compile step - Python is interpreted)
1. Install Python 3.9+
2. In this folder:

       pip install -r requirements.txt
       streamlit run app.py

3. Open http://localhost:8501

## Using your own images
Make a ZIP with one folder per class (e.g. `cats/1.jpg`, `dogs/1.jpg`) and upload it in tab 1.
Without a ZIP, the built-in digits dataset is used.

## Push to GitHub
    git init
    git add .
    git commit -m "SmartVision ML project"
    git branch -M main
    git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
    git push -u origin main
