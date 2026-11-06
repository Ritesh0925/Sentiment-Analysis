# 🌟 Amazon Alexa Review Sentiment Analysis

An end-to-end Machine Learning web application designed to analyze customer reviews for Amazon Alexa devices. Built with **Flask**, **XGBoost**, **Scikit-Learn**, and **NLTK**, the system features both real-time single-sentence sentiment scoring with confidence metrics and batch CSV dataset processing with interactive visual reporting.

---

## 📌 Features

### 1. 💬 Single Review Real-Time Prediction
- **Binary Sentiment Classification**: Predicts whether a review is **Positive** 😊 or **Negative** 😞.
- **Confidence & Probability Meter**: Provides calibrated percentage scores (e.g., `94.2% Positive` vs `5.8% Negative`).
- **NLP Token Insights**: Displays the processed stemmed vocabulary tokens extracted by the NLTK pipeline.
- **Interactive Demo Chips**: 1-click sample reviews to test positive, negative, and mixed scenarios instantly.

### 2. 📁 Bulk CSV & TSV Dataset Processing
- **Drag & Drop Upload**: Modern file upload zone supporting both comma-separated (`.csv`) and tab-separated (`.tsv`) datasets.
- **Smart Column Auto-Detection**: Supports columns named `Sentence`, `review`, `verified_reviews`, `text`, `comment`, or any primary text column automatically.
- **KPI Summary Cards**: Real-time aggregation of **Total Analyzed**, **Positive % & Count**, **Negative % & Count**, and **Dominant Polarity**.
- **Interactive Preview Table**: Shows the first 10 analyzed records with color-coded sentiment tags and confidence scores.
- **Modern Donut Chart**: High-resolution chart visually depicting sentiment proportions.
- **Dual Export (CSV & TSV)**: Download the enriched dataset containing newly predicted sentiment labels and confidence scores in either `.csv` or `.tsv` format.
- **Instant Demo Dataset**: Test bulk analytics with a single click using pre-packaged Alexa review data without uploading a file.

### 3. 🔌 Robust Dual-Mode REST API
- Fully supports JSON payloads (modern web frontend) and form/file multipart data (Streamlit app, cURL, or third-party clients).
- Built-in CORS support and headless Matplotlib rendering for reliable multi-threaded execution.

---

## 🏗️ Project Architecture

```
Sentiment-Analysis-main/
│
├── api.py                            # Flask REST API server & web application entrypoint
├── main.py                           # Streamlit alternative frontend interface
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
│
├── Data/
│   ├── amazon_alexa.tsv              # Original raw Amazon Alexa review dataset
│   ├── SentimentBulk.csv             # Sample CSV for testing bulk review predictions
│   └── Predictions.csv               # Sample exported prediction output
│
├── Models/
│   ├── countVectorizer.pkl           # Trained CountVectorizer vocabulary model
│   ├── scaler.pkl                    # Trained MinMaxScaler model
│   ├── model_xgb.pkl                 # Trained XGBoost classification model
│   ├── model_rf.pkl                  # Trained Random Forest model
│   └── model_dt.pkl                  # Trained Decision Tree model
│
└── templates/
    └── landing.html                  # Modern Tailwind CSS + JavaScript responsive web UI
```

---

## ⚙️ Machine Learning & NLP Pipeline

1. **Text Preprocessing**:
   - Non-alphabet character removal via Regular Expressions.
   - Lowercasing and tokenization.
   - Stopword filtering using NLTK English stopwords dictionary.
   - Porter Stemming to reduce words to their linguistic roots.
2. **Feature Extraction**:
   - Bag-of-Words representation via `CountVectorizer`.
   - Feature scaling via `MinMaxScaler`.
3. **Classification**:
   - `XGBClassifier` calculates class probabilities for final prediction mapping:
     - `0` $\rightarrow$ **Negative**
     - `1` $\rightarrow$ **Positive**

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9 or 3.10
- [Conda](https://docs.conda.io/en/latest/) (recommended) or standard Python virtual environment

### 2. Installation

1. **Clone or navigate to the project directory**:
   ```powershell
   cd d:\projects\Sentiment-Analysis-main
   ```

2. **Create and activate a virtual environment**:
   ```powershell
   conda create -n amazonreview python=3.10 -y
   conda activate amazonreview
   ```

3. **Install the required dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

4. **Verify NLTK stopwords** (automatically downloaded or verify via Python):
   ```python
   python -c "import nltk; nltk.download('stopwords')"
   ```

---

## 🖥️ Running the Application

### Option A: Flask Web Application (Recommended)

Run the Flask server:
```powershell
flask --app api.py run
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### Option B: Streamlit Web Application

To use the alternate Streamlit UI:
1. In one terminal, keep Flask running:
   ```powershell
   flask --app api.py run --port=5000
   ```
2. In a second terminal, start Streamlit:
   ```powershell
   streamlit run main.py
   ```
3. Open the Streamlit URL provided in the terminal (usually `http://localhost:8501`).

---

## 📡 API Reference

### 1. Single Review Prediction
- **Endpoint**: `POST /predict`
- **Headers**: `Content-Type: application/json`
- **Request Body**:
  ```json
  {
    "text": "Love my Echo! The sound quality is amazing and setup took two minutes."
  }
  ```
- **Response**:
  ```json
  {
    "prediction": "Positive",
    "sentiment": "Positive",
    "confidence": 98.4,
    "positive_score": 98.4,
    "negative_score": 1.6,
    "processed_tokens": ["love", "echo", "sound", "qualiti", "amaz", "setup", "took", "two", "minut"],
    "cleaned_text": "love echo sound qualiti amaz setup took two minut"
  }
  ```

### 2. Bulk CSV Dataset Prediction
- **Endpoint**: `POST /predict`
- **Headers**: `Accept: application/json`
- **Body**: `multipart/form-data` with `file: <your_file.csv>` or `<your_file.tsv>`
- **Response**:
  ```json
  {
    "summary": {
      "total": 100,
      "positive_count": 86,
      "positive_pct": 86.0,
      "negative_count": 14,
      "negative_pct": 14.0,
      "text_column_used": "verified_reviews"
    },
    "preview": [
      {
        "id": 1,
        "text": "Love my Echo! Great sound quality.",
        "sentiment": "Positive",
        "confidence": 97.2
      }
    ],
    "graph": "<base64_encoded_png_image>",
    "csv_download": "<base64_encoded_csv_content>",
    "tsv_download": "<base64_encoded_tsv_content>",
    "file_type": "tsv"
  }
  ```

### 3. Demo Dataset Run
- **Endpoint**: `GET /demo-predict`
- Returns instant analysis of the packaged Alexa reviews dataset for quick testing.

### 4. Sample CSV & TSV Downloads
- **Endpoint**: `GET /sample-csv`
  - Downloads a ready-to-use sample CSV template (`sample_reviews.csv`).
- **Endpoint**: `GET /sample-tsv`
  - Downloads the original Amazon Alexa tab-separated template (`sample_reviews.tsv`).

### 5. Health Check
- **Endpoint**: `GET /test`
- Returns service status confirmation.

---

## 📄 Bulk Dataset Format Guide (CSV & TSV)

When uploading your own dataset, your file can be either **comma-separated (`.csv`)** or **tab-separated (`.tsv`)**. The pipeline automatically detects the text column using any of the following standard names:
- `Sentence`
- `verified_reviews` (standard Alexa dataset column)
- `review`
- `text`
- `comment`
- `feedback`

### Example CSV (`.csv`):
```csv
Sentence
Love my Echo! Great sound quality.
Loved it! Best smart speaker ever.
This item did not work. Returning immediately.
Does not work all the time and disconnects.
```

### Example TSV (`.tsv`):
```tsv
rating	date	variation	verified_reviews	feedback
5	31-Jul-18	Charcoal Fabric 	Love my Echo!	1
5	31-Jul-18	Charcoal Fabric 	Loved it!	1
1	31-Jul-18	Sandstone Fabric 	This item did not work. Returning immediately.	0
```

---

## 🛠️ Technologies Used

- **Backend**: Flask, Python 3.10
- **Machine Learning**: XGBoost, Scikit-Learn
- **NLP**: NLTK (Porter Stemmer, Stopwords Corpus)
- **Data & Charts**: Pandas, NumPy, Matplotlib
- **Frontend**: HTML5, Tailwind CSS, Font Awesome, Vanilla JavaScript
- **Alternative UI**: Streamlit

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
