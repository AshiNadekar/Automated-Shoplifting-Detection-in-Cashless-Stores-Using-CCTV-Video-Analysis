# **Automated Shoplifting Detection in Cashless Stores Using CCTV Video Analysis**

## **Overview**
This project implements an **AI-powered shoplifting detection system** using **CCTV video analysis**. It combines **deep learning-based classification** with **real-time person tracking** to identify potential shoplifting incidents in cashless stores. The system is built using **EfficientNetV2B0** for classification and **YOLOv8 with ByteTrack** for person tracking, deployed in a **Streamlit-based application**.

---

## **Dataset**
We use the **DCSASS dataset**, which is publicly available on Kaggle. This dataset consists of **videos categorized into 13 classes**, from which we select the **Shoplifting class**, labeled as **Normal (0) or Shoplifting (1)**. The dataset contains **28 directories**, each holding **31 videos**.

🔗 **Dataset Link:** [DCSASS Dataset on Kaggle](https://www.kaggle.com/datasets/mateohervas/dcsass-dataset)

---

## **System Workflow**

### **1️⃣ Preprocessing & Image Augmentation**
- Extracts frames from videos and **resizes them to 224×224 pixels**.
- Applies **categorical label encoding** (Normal = 0, Shoplifting = 1).
- Performs **data augmentation** to enhance model generalization:
  - **Random Flipping**
  - **Random Rotation**
  - **Random Zoom**
  - **Random Height & Width Scaling**

### **2️⃣ YOLOv8-Based Person Tracking**
- Uses **YOLOv8 from Roboflow** for **real-time person detection**.
- Integrates **Person ByteTracker** to **ensure smooth identity tracking** across frames.

### **3️⃣ EfficientNetV2B0 for Shoplifting Classification**
- Detected person frames from YOLOv8 are **fed into EfficientNetV2B0**.
- The model classifies the activity as **Normal or Shoplifting**.

### **4️⃣ Real-Time Output & Report Generation**
- **Annotated video output** is displayed in **Streamlit UI**.
- **Excel report (CSV format)** is generated with **timestamps and detection results** for further analysis.

---

## **Model Training & Evaluation**
### **Deep Learning Model: EfficientNetV2B0**
- **Pre-trained EfficientNetV2B0** is used as a **feature extractor**.
- **L2 Regularization** prevents overfitting.
- **Binary classification** with **sigmoid activation**.
- **Early stopping** is applied to avoid overfitting.

### **Evaluation Metrics**
| Metric      | Score  |
|------------|--------|
| **Accuracy**  | 78.27% |
| **Precision** | 83.63% |
| **Recall**    | 91.97% |
| **F1-Score**  | 87.60% |

---

## **Installation & Setup**

### **1️⃣ Clone the Repository**
```bash
    git clone https://github.com/your-repo-url.git
    cd your-repo-folder
```

### **2️⃣ Activate Virtual Environment**
#### **For Windows:**
```bash
    env\Scripts\activate
```
#### **For macOS/Linux:**
```bash
    source env/bin/activate
```

### **3️⃣ Install Dependencies**
```bash
    pip install -r requirements.txt
```

### **4️⃣ Run the Streamlit Application**
```bash
    streamlit run main.py
```

---

## **Required Dependencies**
Ensure the following dependencies are listed in **requirements.txt**:
```txt
inference-sdk
numpy
pandas
opencv-python
tensorflow==2.15.1
supervision
streamlit
```

