import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import lightgbm as lgb

from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

global IMG_SIZE, MODEL_DIR, RESULTS_DIR, categories
IMG_SIZE = (64, 64)
MODEL_DIR = "models"
RESULTS_DIR = "results"

def uploadDataset():
    global filename, categories
    filename = filedialog.askdirectory(initialdir="Dataset")
    if not filename:
        return

    text.delete('1.0', END)
    text.insert(END, f"Folder Loaded:\n{filename}\n\n")

    categories = [
        d for d in os.listdir(filename)
        if os.path.isdir(os.path.join(filename, d))
    ]

    text.insert(END, "Subfolders found:\n")
    for label in categories:
        text.insert(END, f"- {label}\n")

def extract_features(img):
    img = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    pixels = gray.flatten().astype(np.float32)

    # Statistical features
    mean_val = np.mean(pixels)  # Average brightness of the image
    std_val = np.std(pixels)  # Measures contrast
    skew_val = skew(pixels)  # Measures asymmetry of pixel intensity distribution
    kurt_val = kurtosis(pixels) # Measures tailedness of distribution (extreme pixel or not)
    energy_val = np.mean(pixels ** 2) # Measures overall intensity strength

    # GLCM
    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm, 'contrast').mean() # Measures local intensity variation
    homogeneity = graycoprops(glcm, 'homogeneity').mean()  # Measures closeness of pixel pairs
    correlation = graycoprops(glcm, 'correlation').mean() # Measures how correlated a pixel is to its neighbor
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean() # Similar to contrast, but linear
    asm = graycoprops(glcm, 'ASM').mean() # Measures uniformity

    return np.array([
        mean_val, std_val, skew_val, kurt_val,
        energy_val, contrast, dissimilarity,
        homogeneity, correlation, asm
    ], dtype=np.float32)

def preprocessing():
    global  X, Y
    global X
    X_file = os.path.join(MODEL_DIR, "X.txt.npy")
    Y_file = os.path.join(MODEL_DIR, "Y.txt.npy")

    if os.path.exists(X_file) and os.path.exists(Y_file):
        X = np.load(X_file)
        Y = np.load(Y_file)
        text.insert(END, "Loaded cached features.\n\n")
    else:
        X, Y = [], []

        for root, _, files in os.walk(filename):
            label = os.path.basename(root)
            if label not in categories:
                continue

            for file in files:
                if file.lower().endswith(('jpg','png','jpeg')):
                    print(root)
                    img = cv2.imread(os.path.join(root, file))
                    features = extract_features(img)
                    X.append(features)
                    Y.append(categories.index(label))

        X = np.array(X)
        Y = np.array(Y)
        np.save(X_file, X)
        np.save(Y_file, Y)

    text.insert(END, f"Feature matrix: {X.shape}\n\n")

def train():
    global x_train, x_test, y_train, y_test, X

    # Replace NaNs with column mean
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Now SMOTE will work
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=77, stratify=Y
    )

    from imblearn.over_sampling import SMOTE
    sm = SMOTE()
    x_train, y_train = sm.fit_resample(x_train, y_train)

    text.insert(END, f"X_train: {x_train.shape}\n\n")
    text.insert(END, f"y_train: {y_train.shape}\n\n")



# ──────────────────────────────────────────────────────────────
# Global containers
# ──────────────────────────────────────────────────────────────

global metrics_overall, class_metrics_storage
metrics_overall = []                    # Overall metrics
class_metrics_storage = {}              # Key = class name → list of dicts (one per model)
def calculateMetrics(algorithm, predict, testY):
    
    testY = testY.astype('int')
    predict = predict.astype('int')
    
    # ── Overall metrics ─────────────────────────────────────
    acc  = accuracy_score(testY, predict) * 100
    prec = precision_score(testY, predict, average='macro', zero_division=0) * 100
    rec  = recall_score(testY, predict, average='macro', zero_division=0) * 100
    f1   = f1_score(testY, predict, average='macro', zero_division=0) * 100
    
    metrics_overall.append({
        'Model': algorithm,
        'Accuracy':  round(acc, 2),
        'Precision': round(prec, 2),
        'Recall':    round(rec, 2),
        'F1-Score':  round(f1, 2)
    })
    
    text.insert(END, f"\n=== {algorithm} Overall ===\n")
    text.insert(END, f"Accuracy  : {acc:.2f}%\n")
    text.insert(END, f"Precision : {prec:.2f}%\n")
    text.insert(END, f"Recall    : {rec:.2f}%\n")
    text.insert(END, f"F1-Score  : {f1:.2f}%\n")
    
    # ── Class-wise metrics ──────────────────────────────────
    report = classification_report(testY, predict, target_names=categories,
                                  output_dict=True, zero_division=0)
    text.insert(END, "\n=== Classification Report ===")
    text.insert(END, classification_report(testY, predict, target_names=categories, zero_division=0))

    for cls in categories:
        # Initialize list for this class if first time
        if cls not in class_metrics_storage:
            class_metrics_storage[cls] = []
        
        class_metrics_storage[cls].append({
            'Model':     algorithm,
            'Precision': round(report[cls]['precision'] * 100, 2),
            'Recall':    round(report[cls]['recall'] * 100, 2),
            'F1-Score':  round(report[cls]['f1-score'] * 100, 2)
        })
    
    # ── Confusion matrix ────────────────────────────────────
    cm = confusion_matrix(testY, predict)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=categories, yticklabels=categories, cbar=False)
    plt.title(f"{algorithm} - Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def model1():
    global model_path, rid_clf, predict
    from sklearn.linear_model import RidgeClassifier
    import os
    import joblib

    model_path = r"models\ridge_classifier.pkl"

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        rid_clf = joblib.load(model_path)
        predict = rid_clf.predict(x_test)
        calculateMetrics("Ridge Classifier", predict, y_test)

    else:
        rid_clf = RidgeClassifier(alpha=1.0) #initailize
        rid_clf.fit(x_train, y_train) #train
        predict = rid_clf.predict(x_test)
        joblib.dump(rid_clf, model_path)
        calculateMetrics("Ridge Classifier", predict, y_test)

def model2():
    global model_path, qda_clf, predict
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    import os
    import joblib

    model_path = r"models\qda_classifier.pkl"

    # Ensure folder exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        qda_clf = joblib.load(model_path)
        predict = qda_clf.predict(x_test)
        calculateMetrics("QDA Classifier", predict, y_test)

    else:
        qda_clf = QuadraticDiscriminantAnalysis()
        qda_clf.fit(x_train, y_train)
        predict = qda_clf.predict(x_test)
        joblib.dump(qda_clf, model_path)
        calculateMetrics("QDA Classifier", predict, y_test)


def model3():
    global model_path, knn_clf, predict
    from sklearn.neighbors import KNeighborsClassifier
    import os
    import joblib

    model_path = r"models\knn_classifier.pkl"

    # Ensure folder exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        knn_clf = joblib.load(model_path)
        predict = knn_clf.predict(x_test)
        calculateMetrics("KNN Classifier", predict, y_test)

    else:
        knn_clf = KNeighborsClassifier(
            n_neighbors=5,
            metric="minkowski",  # default (Euclidean when p=2)
            p=2
        )
        knn_clf.fit(x_train, y_train)
        predict = knn_clf.predict(x_test)
        joblib.dump(knn_clf, model_path)
        calculateMetrics("KNN Classifier", predict, y_test)



def model4():
    global model_path, xgb_clf, predict
    from sklearn.ensemble import ExtraTreesClassifier
    import os
    import joblib

    model_path = r"models\extra_trees_classifier.pkl"

    # Ensure folder exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        et_clf = joblib.load(model_path)
        predict = et_clf.predict(x_test)
        calculateMetrics("Extra Trees Classifier", predict, y_test)

    else:
        et_clf = ExtraTreesClassifier(n_estimators=100,random_state=42,n_jobs=-1)
        et_clf.fit(x_train, y_train)
        predict = et_clf.predict(x_test)
        joblib.dump(et_clf, model_path)
        calculateMetrics("Extra Trees Classifier", predict, y_test)


# ----------------------------
# Prediction function
# ----------------------------
def predict_image():

    global model_path, pred_idx, pred_label, features, predicted_class, model
    model_path = os.path.join(MODEL_DIR, "extra_trees_classifier.pkl")
    if not os.path.exists(model_path):
        messagebox.showwarning("Warning", "Train model first!")
        return

    model = joblib.load(model_path)

    path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if not path:
        return

    # Load image safely
    img_array = cv2.imread(path)
    if img_array is None:
        print(f"Error: Cannot read the image: {path}")
        return None

    # Extract features (resized to IMG_SIZE)
    features = extract_features(img_array).reshape(1, -1)

    # Safety check
    if features.shape[1] != model.n_features_in_:
        raise ValueError(
            f"Feature vector length {features.shape[1]} does not match "
            f"model expected {model.n_features_in_} features."
        )

    # Predict
    pred_idx = model.predict(features)[0]
    pred_label = categories[pred_idx]

    # Display image with predicted label
    plt.figure(figsize=(5,5))
    plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    plt.text(10, 10, f'Predicted: {pred_label}', color='yellow', fontsize=12,
             weight='bold', backgroundcolor='white')
    plt.axis('off')
    plt.show()

    return pred_label

#predicted_class = predict_image(model_path, categories)
#text.insert(END, "Predicted Class:", predicted_class)




# ================= UI =================

main = Tk()
main.geometry("1400x900")
bg_image = Image.open("surface_defect_detection_under_solution_banner_image.webp")   
bg_image = bg_image.resize((1400, 900))
bg_photo = ImageTk.PhotoImage(bg_image)

bg_label = Label(main, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

font = ('times', 15, 'bold')
font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

Label(main, text="Surface Defects from Multi Domain Inspection Images",
      bg="lightyellow", fg="black",
      font=font, height=3, width=120).place(x=0, y=5)

Button(main, text="Dataset", command=uploadDataset, font=ff).place(x=20, y=150)
Button(main, text="Feature Extraction", command=preprocessing, font=ff).place(x=20, y=200)
Button(main, text="Train Test Split", command=train, font=ff).place(x=20, y=250)
Button(main, text="Ridge Classifier", command=model1, font=ff).place(x=20, y=300)
Button(main, text="QDA Classifier", command=model2, font=ff).place(x=20, y=350)
Button(main, text="KNN Classifier", command=model3, font=ff).place(x=20, y=400)
Button(main, text="Extra Trees Classifier", command=model4, font=ff).place(x=20, y=450)
Button(main, text="Prediction", command=predict_image, font=ff).place(x=20, y=500)

text = Text(main, height=25, width=100, font=font1)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=330, y=100)

main.mainloop()
