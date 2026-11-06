from flask import Flask, request, jsonify, send_file, render_template, Response
import re
from io import BytesIO
import os
import pickle
import base64
import warnings

# Use headless/non-interactive backend for matplotlib in web servers to prevent GUI thread warnings and crashes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Suppress version mismatch warnings when unpickling older models
warnings.filterwarnings("ignore", category=UserWarning)
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd

STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)

# Base directory for reliable model path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "Models")

# Load models once at startup rather than on every HTTP request
with open(os.path.join(MODELS_DIR, "model_xgb.pkl"), "rb") as f:
    predictor = pickle.load(f)
with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(MODELS_DIR, "countVectorizer.pkl"), "rb") as f:
    cv = pickle.load(f)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept"
    response.headers["Access-Control-Expose-Headers"] = "X-Graph-Exists, X-Graph-Data"
    return response


@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."


@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("landing.html")


@app.route("/sample-csv", methods=["GET"])
def sample_csv():
    """Serves a sample CSV file so users can test bulk predictions easily."""
    sample_path = os.path.join(BASE_DIR, "Data", "SentimentBulk.csv")
    if os.path.exists(sample_path):
        return send_file(
            sample_path,
            mimetype="text/csv",
            as_attachment=True,
            download_name="sample_reviews.csv",
        )
    sample_content = (
        "Sentence\n"
        "Love my Echo! Great sound quality.\n"
        "Loved it! Voice recognition works wonders.\n"
        "This item did not work. Returning immediately.\n"
        "Does not work all the time, very frustrating.\n"
        "Super easy to setup and very convenient for daily use.\n"
    )
    return Response(
        sample_content,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=sample_reviews.csv"},
    )


@app.route("/sample-tsv", methods=["GET"])
def sample_tsv():
    """Serves a sample TSV file (Tab-Separated Values)."""
    sample_path = os.path.join(BASE_DIR, "Data", "amazon_alexa.tsv")
    if os.path.exists(sample_path):
        return send_file(
            sample_path,
            mimetype="text/tab-separated-values",
            as_attachment=True,
            download_name="sample_reviews.tsv",
        )
    sample_content = (
        "rating\tdate\tvariation\tverified_reviews\tfeedback\n"
        "5\t31-Jul-18\tCharcoal Fabric \tLove my Echo!\t1\n"
        "5\t31-Jul-18\tCharcoal Fabric \tLoved it!\t1\n"
        "1\t31-Jul-18\tSandstone Fabric \tThis item did not work. Returning immediately.\t0\n"
        "2\t31-Jul-18\tHeather Gray Fabric \tDoes not work all the time.\t0\n"
        "5\t31-Jul-18\tOak Finish \tSuper easy to setup and works great.\t1\n"
    )
    return Response(
        sample_content,
        mimetype="text/tab-separated-values",
        headers={"Content-Disposition": "attachment; filename=sample_reviews.tsv"},
    )


@app.route("/demo-predict", methods=["GET"])
def demo_predict():
    """Runs a live demo prediction using sample data for immediate web preview."""
    try:
        sample_tsv_path = os.path.join(BASE_DIR, "Data", "amazon_alexa.tsv")
        if os.path.exists(sample_tsv_path):
            data = pd.read_csv(sample_tsv_path, sep="\t").head(25)
        else:
            sample_csv_path = os.path.join(BASE_DIR, "Data", "SentimentBulk.csv")
            if os.path.exists(sample_csv_path):
                data = pd.read_csv(sample_csv_path)
            else:
                data = pd.DataFrame(
                    {
                        "Sentence": [
                            "Love my Echo! Great sound quality.",
                            "Loved it! Best smart speaker ever.",
                            "This item did not work. Returning immediately.",
                            "Does not work all the time and disconnects.",
                            "Great value for money, highly recommended.",
                        ]
                    }
                )

        predictions_csv, predictions_tsv, graph, summary, preview = bulk_prediction(
            predictor, scaler, cv, data
        )
        b64_graph = base64.b64encode(graph.getvalue()).decode("ascii")
        csv_b64 = base64.b64encode(predictions_csv.getvalue()).decode("ascii")
        tsv_b64 = base64.b64encode(predictions_tsv.getvalue()).decode("ascii")

        return jsonify(
            {
                "summary": summary,
                "preview": preview,
                "graph": b64_graph,
                "csv_download": csv_b64,
                "tsv_download": tsv_b64,
                "file_type": "tsv",
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if the request contains a file (for bulk prediction)
        if "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"error": "No file selected."}), 400

            filename = file.filename.lower()
            if not (filename.endswith(".csv") or filename.endswith(".tsv") or filename.endswith(".txt")):
                return jsonify({"error": "Unsupported file format. Please upload a .csv or .tsv file."}), 400

            # Detect separator based on extension or python sniffer
            try:
                if filename.endswith(".tsv"):
                    data = pd.read_csv(file, sep="\t")
                else:
                    try:
                        data = pd.read_csv(file, sep=None, engine="python")
                    except Exception:
                        file.seek(0)
                        data = pd.read_csv(file)
            except Exception as read_err:
                return jsonify({"error": f"Failed to parse file: {str(read_err)}"}), 400

            if data.empty:
                return jsonify({"error": "The uploaded file contains no data."}), 400

            predictions_csv, predictions_tsv, graph, summary, preview = bulk_prediction(
                predictor, scaler, cv, data
            )
            b64_graph = base64.b64encode(graph.getvalue()).decode("ascii")

            # Check if JSON format requested (by modern web client or explicit flag)
            wants_json = (
                request.headers.get("Accept") == "application/json"
                or request.form.get("format") == "json"
                or request.args.get("format") == "json"
            )

            is_tsv = filename.endswith(".tsv")

            if wants_json:
                csv_b64 = base64.b64encode(predictions_csv.getvalue()).decode("ascii")
                tsv_b64 = base64.b64encode(predictions_tsv.getvalue()).decode("ascii")
                return jsonify(
                    {
                        "summary": summary,
                        "preview": preview,
                        "graph": b64_graph,
                        "csv_download": csv_b64,
                        "tsv_download": tsv_b64,
                        "file_type": "tsv" if is_tsv else "csv",
                    }
                )

            # Standard download response for Streamlit or simple clients
            download_buf = predictions_tsv if is_tsv else predictions_csv
            mimetype = "text/tab-separated-values" if is_tsv else "text/csv"
            download_name = "Predictions.tsv" if is_tsv else "Predictions.csv"

            response = send_file(
                download_buf,
                mimetype=mimetype,
                as_attachment=True,
                download_name=download_name,
            )
            response.headers["X-Graph-Exists"] = "true"
            response.headers["X-Graph-Data"] = b64_graph
            return response

        # Single string prediction (supports JSON payload or Form data)
        text_input = None
        if request.is_json and request.json and "text" in request.json:
            text_input = request.json["text"]
        elif request.form and "text" in request.form:
            text_input = request.form["text"]

        if text_input is not None:
            if not str(text_input).strip():
                return jsonify({"error": "Please enter some text to analyze."}), 400

            result = single_prediction(predictor, scaler, cv, str(text_input))
            return jsonify(
                {
                    "prediction": result["sentiment"],
                    "sentiment": result["sentiment"],
                    "confidence": result["confidence"],
                    "positive_score": result["positive_score"],
                    "negative_score": result["negative_score"],
                    "processed_tokens": result["processed_tokens"],
                    "cleaned_text": result["cleaned_text"],
                }
            )

        return jsonify({"error": "No valid file or text provided."}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def single_prediction(predictor, scaler, cv, text_input):
    corpus = []
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    words = review.lower().split()
    stemmed_words = [stemmer.stem(word) for word in words if word not in STOPWORDS]
    cleaned_review = " ".join(stemmed_words)
    corpus.append(cleaned_review)

    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    probabilities = predictor.predict_proba(X_prediction_scl)[0]

    # Model classes: 0 -> Negative, 1 -> Positive
    neg_score = round(float(probabilities[0]) * 100, 1)
    pos_score = round(float(probabilities[1]) * 100, 1)

    prediction_idx = probabilities.argmax()
    sentiment = "Positive" if prediction_idx == 1 else "Negative"
    confidence = pos_score if sentiment == "Positive" else neg_score

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "positive_score": pos_score,
        "negative_score": neg_score,
        "processed_tokens": stemmed_words,
        "cleaned_text": cleaned_review,
    }


def find_text_column(df):
    """Auto-detects the text/review column even if column names differ across datasets."""
    candidates = [
        "sentence",
        "review",
        "verified_reviews",
        "text",
        "comment",
        "feedback",
        "body",
        "content",
    ]
    col_map = {str(c).lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in col_map:
            return col_map[cand]

    # Fallback to the first object or string column
    for c in df.columns:
        if df[c].dtype == object:
            return c
    return df.columns[0]


def bulk_prediction(predictor, scaler, cv, data):
    col = find_text_column(data)
    corpus = []
    stemmer = PorterStemmer()
    for i in range(0, data.shape[0]):
        raw_val = data.iloc[i][col]
        text_val = str(raw_val) if pd.notna(raw_val) else ""
        review = re.sub("[^a-zA-Z]", " ", text_val)
        words = review.lower().split()
        stemmed_words = [stemmer.stem(w) for w in words if w not in STOPWORDS]
        corpus.append(" ".join(stemmed_words))

    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    probabilities = predictor.predict_proba(X_prediction_scl)
    y_predictions = probabilities.argmax(axis=1)

    confidences = [
        round(float(probabilities[i][y_predictions[i]]) * 100, 1)
        for i in range(len(y_predictions))
    ]
    sentiments = list(map(sentiment_mapping, y_predictions))

    data["Predicted sentiment"] = sentiments
    data["Confidence (%)"] = confidences

    total = len(sentiments)
    pos_count = sentiments.count("Positive")
    neg_count = sentiments.count("Negative")
    pos_pct = round((pos_count / total * 100), 1) if total > 0 else 0
    neg_pct = round((neg_count / total * 100), 1) if total > 0 else 0

    summary = {
        "total": total,
        "positive_count": pos_count,
        "positive_pct": pos_pct,
        "negative_count": neg_count,
        "negative_pct": neg_pct,
        "text_column_used": col,
    }

    # First 10 rows for rich table preview
    preview = []
    for i in range(min(10, total)):
        raw_snippet = str(data.iloc[i][col]) if pd.notna(data.iloc[i][col]) else ""
        if len(raw_snippet) > 130:
            raw_snippet = raw_snippet[:127] + "..."
        preview.append(
            {
                "id": i + 1,
                "text": raw_snippet,
                "sentiment": sentiments[i],
                "confidence": confidences[i],
            }
        )

    # Generate CSV output
    predictions_csv = BytesIO()
    data.to_csv(predictions_csv, index=False, sep=",")
    predictions_csv.seek(0)

    # Generate TSV output
    predictions_tsv = BytesIO()
    data.to_csv(predictions_tsv, index=False, sep="\t")
    predictions_tsv.seek(0)

    graph = get_distribution_graph(data)

    return predictions_csv, predictions_tsv, graph, summary, preview


def get_distribution_graph(data):
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")

    counts = data["Predicted sentiment"].value_counts()
    color_map = {"Positive": "#10B981", "Negative": "#EF4444"}
    colors = [color_map.get(k, "#6B7280") for k in counts.index]

    wedges, texts, autotexts = ax.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        pctdistance=0.75,
        explode=[0.02] * len(counts) if len(counts) > 1 else None,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2.5),
        textprops=dict(fontsize=11, fontweight="bold", color="#1F2937"),
    )

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(11)

    ax.set_title(
        "Sentiment Breakdown",
        fontsize=14,
        fontweight="bold",
        pad=15,
        color="#111827",
    )

    total = len(data)
    ax.text(
        0,
        0,
        f"{total}\nReviews",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#374151",
    )

    plt.tight_layout()
    graph = BytesIO()
    plt.savefig(graph, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    graph.seek(0)

    return graph


def sentiment_mapping(x):
    return "Positive" if x == 1 else "Negative"


if __name__ == "__main__":
    app.run(port=5000, debug=True)
