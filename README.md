# IMDb Movie Review Sentiment Classifier

Binary sentiment classification system benchmarked on the IMDb dataset (50,000 labeled reviews: 25,000 train / 25,000 test). The implementation evaluates three distinct natural language processing representation paradigms: sparse term frequencies (TF-IDF), static pretrained distributed vectors (GloVe), and contextual transformer architectures (Sentence-BERT and end-to-end fine-tuned DistilBERT).

Downstream classifiers are standardized using Logistic Regression across feature extraction methods to isolate the representation quality from classifier complexity, followed by full fine-tuning and production model serialization.

---

## Architecture and Workflow

### 1. Data Ingestion and Statistical Exploration

* **Dataset:** Stanford IMDb Large Movie Review Dataset containing 50,000 reviews balanced equally with 50% positive and 50% negative labels across both training and test splits.
* **Validation Partition:** A stratified 80/20 train-validation split is created from the training set, yielding 20,000 training samples and 5,000 validation samples.
* **Sequence Length Analysis:** Analysis of review word counts shows a median length of 174 words, a mean of 233.8 words, and a 95th percentile of 598 words. Setting the transformer sequence length to 256 tokens truncates approximately 26.8% of reviews, introducing an empirical trade-off between throughput and document tail preservation.

### 2. Dual-Track Text Preprocessing

To maximize performance across fundamentally different tokenization algorithms, two dedicated cleaning tracks are implemented:

* **Track A (Bag-of-Words and Static Embeddings):** Lowercasing, HTML tag stripping via regex, removal of punctuation and numerical digits (`[^a-z\s]`), and whitespace normalization. This eliminates vocabulary fragmentation for TF-IDF and GloVe.
* **Track B (Contextual Transformers):** HTML tag removal and whitespace normalization only. Casing, punctuation, and contractions are preserved because WordPiece subword tokenization and self-attention mechanisms rely on these cues to parse syntactic dependencies and negation boundaries.

### 3. Feature Extraction and Modeling Methods

* **Method 1: TF-IDF + Logistic Regression:** Sparse n-gram representation using unigrams and bigrams up to 50,000 features with sublinear term frequency scaling (`1 + log(tf)`). Regularization strength is tuned across validation candidates `C in [0.1, 1.0, 10.0]`, selecting `C = 1.0` as optimal.
* **Method 2: GloVe Pretrained Embeddings + Logistic Regression:** 100-dimensional word vectors pretrained on 6 billion Wikipedia and Gigaword tokens. Each review is mapped to a dense 100-dimensional vector via global mean-pooling. Out-of-vocabulary tracking measures missed tokens and types.
* **Method 3: Sentence-BERT Embeddings + Logistic Regression:** 384-dimensional dense contextual sentence representations extracted using `all-MiniLM-L6-v2`. A downstream Logistic Regression classifier is fit on a stratified 5,000-review subset.
* **Method 4: Fine-Tuned DistilBERT:** End-to-end classification fine-tuning of `distilbert-base-uncased` with a linear classification head using PyTorch and Hugging Face Trainer. Trained for 2 epochs on GPU with mixed precision (`fp16`), warmup scheduling, weight decay, and automatic checkpoint restoration of the lowest validation loss state.

### 4. Controlled Apples-to-Apples Benchmark

Because frozen BERT and DistilBERT were trained on 5,000 samples due to compute constraints, TF-IDF and GloVe are re-evaluated on the exact same 5,000 training and 2,500 test samples (indexed identically via `StratifiedShuffleSplit`). This eliminates sample-size disparity and measures true feature quality.

### 5. Production Serialization and Pipeline Verification

The winning DistilBERT model weights and tokenizer are exported via `save_pretrained`. The classical TF-IDF vectorizer and classifier are serialized via `joblib.dump`. Both pipelines are reloaded from disk to verify inference on unseen production reviews.

---

## Tech Stack

Below is the breakdown of each technology, library, and specific module utilized, detailing its exact role in the pipeline:

### 1. Core Programming Environment

* **Python 3.10+:** Primary runtime environment for data orchestration, feature extraction, model training, and production serialization.

### 2. Deep Learning Framework

* **PyTorch (`torch`):**
  * Provides the computational backend for DistilBERT tensor manipulations.
  * Enables GPU acceleration (`torch.cuda.is_available()`) and 16-bit mixed precision (`fp16=True`) for 2x faster training throughput with reduced VRAM usage.
  * Powers tensor wrapping in `IMDbDataset(Dataset)` and gradient backpropagation across transformer layers.

### 3. Transformers and Natural Language Processing

* **Hugging Face Transformers (`transformers`):**
  * `DistilBertTokenizerFast`: Implements fast Rust-backed subword (WordPiece) tokenization. Converts raw text into input token IDs and attention masks with automated truncation (`max_length=256`) and padding.
  * `DistilBertForSequenceClassification`: Instantiates a pretrained 6-layer DistilBERT base model with a specialized 2-class linear classification head placed on top of the pooled transformer output.
  * `Trainer` and `TrainingArguments`: Manages the complete training lifecycle, including learning rate scheduling, warmup steps (100 steps), weight decay (0.01), evaluation every epoch, and automatic checkpoint restoration for the best validation loss state.
* **Hugging Face Datasets (`datasets`):**
  * `load_dataset('stanfordnlp/imdb')`: Streams and downloads the benchmark 50,000 IMDb movie reviews into memory-efficient columnar Arrow tables, separated into 25k train and 25k test splits.
* **Sentence-Transformers (`sentence-transformers`):**
  * `SentenceTransformer('all-MiniLM-L6-v2')`: Extracted 384-dimensional dense contextual sentence representations. Used to evaluate the effectiveness of frozen transformer embeddings before full fine-tuning.
* **Gensim (`gensim.downloader`):**
  * `gensim.downloader.load('glove-wiki-gigaword-100')`: Fetches the 400,000-word GloVe lookup table pretrained on 6 billion tokens. Maps word tokens to static 100-dimensional distributed vectors for semantic mean-pooling.

### 4. Classical Machine Learning and Metrics

* **Scikit-Learn (`sklearn`):**
  * `TfidfVectorizer`: Transforms cleaned reviews into a high-dimensional sparse matrix of 50,000 unigrams and bigrams with sublinear term frequency scaling (`1 + log(tf)`).
  * `LogisticRegression`: The shared downstream linear classifier used across TF-IDF, GloVe, and Sentence-BERT. Keeping this model constant isolates representation quality from classifier capacity.
  * `train_test_split` & `StratifiedShuffleSplit`: Guarantees exact 50/50 positive/negative class balance across train, validation, test, and the equal 5,000-sample benchmark partition.
  * `accuracy_score`, `precision_score`, `recall_score`, `f1_score`: Computes full macro-averaged classification performance across all models.
  * `roc_auc_score` & `roc_curve`: Computes threshold-independent Area Under the Receiver Operating Characteristic curve and coordinates for multi-model diagnostic plots.
  * `confusion_matrix` & `classification_report`: Generates class-level true positive, true negative, false positive, and false negative counts.

### 5. Numerical Processing and Data Analytics

* **NumPy (`numpy`):**
  * Performs fast vector calculations, including row-wise mean-pooling (`np.mean(vecs, axis=0)`) across GloVe word vectors to construct review embeddings.
  * Computes statistical metrics such as median, mean, and 95th percentile lengths, and converts model logit arrays via softmax operations.
* **Pandas (`pandas`):**
  * Structures review splits and labels into DataFrames for statistical indexing.
  * Formats the unified Model Comparison Summary table and coordinates live inference outputs for side-by-side model predictions.

### 6. Data Visualization

* **Matplotlib (`matplotlib.pyplot`):**
  * Orchestrates multi-subplot canvas layouts, custom bar width positioning, legend styling, and axis range constraints.
  * Generates the multi-model ROC curve plot (True Positive Rate vs False Positive Rate) and the side-by-side four-metric grouped bar chart.
* **Seaborn (`seaborn`):**
  * Renders color-coded confusion matrix heatmaps using distinct monochromatic palettes (`Blues`, `Greens`, `Oranges`, `Purples`).
  * Generates the Kernel Density Estimation (KDE) review word count distribution plot with overlay thresholds.

### 7. Production Model Serialization

* **Joblib (`joblib`):**
  * Serializes the classical pipeline (`TfidfVectorizer` + `LogisticRegression`) into a standalone binary file (`tfidf_pipeline.joblib`) for sub-millisecond CPU deployment without deep learning dependencies.
* **Hugging Face `save_pretrained`:**
  * Exports the fine-tuned DistilBERT weights (`model.safetensors`), configuration dictionary (`config.json`), and subword vocabulary (`vocab.txt`) into a standardized folder for microservice serving.

---

## Directory Structure

```text
IMDb Movie Review Sentiment Classifier/
|-- .gitignore                                     # Git ignore patterns (model weights, cache)
|-- IMDb Movie Review Sentiment Classifier.ipynb   # Complete executable notebook (51 cells)
|-- LICENSE                                        # Repository license
|-- README.md                                      # Documentation and benchmark report
\-- saved_models/                                  # Exported production model artifacts (generated via Section 10)
    |-- distilbert_sentiment/                      # Exported DistilBERT weights and tokenizer
    |   |-- config.json
    |   |-- model.safetensors
    |   |-- tokenizer_config.json
    |   \-- vocab.txt
    \-- tfidf_pipeline.joblib                      # Serialized TF-IDF vectorizer and classifier
```

---

## How to Run the Project

### 1. Environment Setup

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/mdtausifjafar/IMDb-Movie-Review-Sentiment-Classifier.git
cd "IMDb Movie Review Sentiment Classifier"
pip install torch transformers datasets sentence-transformers gensim scikit-learn pandas numpy matplotlib seaborn joblib
```

### 2. Running the Complete Experiment

The end-to-end benchmark is fully contained in `IMDb Movie Review Sentiment Classifier.ipynb`:

* **Google Colab (Recommended):**
  1. Upload `IMDb Movie Review Sentiment Classifier.ipynb` to Google Colab.
  2. Navigate to `Runtime > Change runtime type` and select **T4 GPU** for fast transformer fine-tuning.
  3. Run all cells (`Runtime > Run all`).
* **Local Jupyter Environment:**
  Launch Jupyter Lab or VS Code and execute the notebook cells sequentially from top to bottom.

---

## How to Run Inference with Saved Models

Once Section 10 of the notebook is executed, both production models are saved into the `saved_models/` directory. You can reload and use either model in Python with the scripts below:

### Option A: Inference with Fine-Tuned DistilBERT (High Accuracy)

```python
import torch
import numpy as np
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

# Load exported model and tokenizer from disk
model_path = "./saved_models/distilbert_sentiment"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Input text to classify
review = "An extraordinary cinematic achievement with masterful acting and direction."

# Tokenize and run forward pass
inputs = tokenizer(review, truncation=True, padding=True, max_length=256, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].numpy()
    pred = int(np.argmax(probs))

sentiment = "Positive" if pred == 1 else "Negative"
confidence = float(probs[pred]) * 100

print(f"Review    : {review}")
print(f"Sentiment : {sentiment} (Confidence: {confidence:.2f}%)")
```

### Option B: Inference with TF-IDF Baseline (Fast CPU Serving)

```python
import re
import joblib

# Minimal text cleaner
def preprocess(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

# Load serialized TF-IDF pipeline from disk
pipeline = joblib.load("./saved_models/tfidf_pipeline.joblib")
vectorizer = pipeline["vectorizer"]
classifier = pipeline["classifier"]

# Input text to classify
review = "An extraordinary cinematic achievement with masterful acting and direction."
clean_text = preprocess(review)
vec = vectorizer.transform([clean_text])
pred = int(classifier.predict(vec)[0])
conf = float(classifier.predict_proba(vec)[0][pred]) * 100

sentiment = "Positive" if pred == 1 else "Negative"
print(f"Review    : {review}")
print(f"Sentiment : {sentiment} (Confidence: {conf:.2f}%)")
```

---

## About the Saved Model Files

Running Section 10 of the notebook automatically creates and populates the `saved_models/` directory:

1. **`./saved_models/distilbert_sentiment/` (Deep Learning Tier):**
   * **`model.safetensors` (~268 MB):** Serialized transformer weights in Hugging Face safetensors format, preventing arbitrary code execution during deserialization.
   * **`config.json`:** Model configuration defining architecture dimensions (6 layers, 768 hidden size, 12 attention heads) and binary output label mapping.
   * **`tokenizer_config.json` & `vocab.txt`:** WordPiece vocabulary (30,522 subword tokens) and special token formatting (`[CLS]`, `[SEP]`, `[PAD]`).
2. **`./saved_models/tfidf_pipeline.joblib` (~1.5 MB) (CPU Fallback Tier):**
   * A serialized Python dictionary containing the fitted `TfidfVectorizer` (50,000 vocabulary n-grams) and the trained `LogisticRegression` weight coefficients for instantaneous, zero-GPU inference.

### Why Saved Models are in `.gitignore`

In accordance with machine learning repository standards, the `saved_models/` folder is excluded from version control via `.gitignore`. The DistilBERT weights file (`model.safetensors`) is 268 MB, which exceeds GitHub's 100 MB single-file limit. Running Section 10 of the notebook reproduces these exact files locally on demand.

---

## Empirical Benchmark Results

### Full Dataset Evaluation Summary

The table below summarizes performance across the full evaluation partitions. Note that TF-IDF and GloVe trained their downstream classifiers on 20,000 reviews, whereas BERT variants used a 5,000-sample training subset due to GPU compute limits.

| Method                                        | Train Samples | Test Samples | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Total Time (s) |
| :-------------------------------------------- | :-----------: | :----------: | :------: | :-------: | :----: | :------: | :-----: | :------------: |
| **TF-IDF + Logistic Regression**        |    20,000    |    25,000    |  0.8936  |  0.8937  | 0.8936 |  0.8936  | 0.9606 |     22.43     |
| **GloVe + Logistic Regression**         |    20,000    |    25,000    |  0.7973  |  0.7974  | 0.7973 |  0.7973  | 0.8768 |     19.87     |
| **BERT (frozen) + Logistic Regression** |     5,000     |    2,500    |  0.7972  |  0.7972  | 0.7972 |  0.7972  | 0.8848 |     22.89     |
| **DistilBERT (fine-tuned)**             |     5,000     |    2,500    |  0.8948  |  0.8948  | 0.8948 |  0.8948  | 0.9628 |     101.44     |

### Controlled Apples-to-Apples Benchmark (Equal 5,000 Train / 2,500 Test)

When trained and evaluated on the exact same review samples, contextual representation learning demonstrates clear superiority over bag-of-words and static embeddings:

| Model Architecture                                                | Training Size | Test Size |  Test Accuracy  |
| :---------------------------------------------------------------- | :-----------: | :-------: | :--------------: |
| **GloVe (100d mean-pool) + Logistic Regression**            |     5,000     |   2,500   |      0.7892      |
| **Sentence-BERT (384d frozen) + Logistic Regression**       |     5,000     |   2,500   |      0.7972      |
| **TF-IDF (1-2 ngrams, 50k features) + Logistic Regression** |     5,000     |   2,500   |      0.8716      |
| **Fine-Tuned DistilBERT (end-to-end)**                      |     5,000     |   2,500   | **0.8948** |

---

## Out-of-Vocabulary (OOV) Analysis for GloVe

Analysis of GloVe 100d coverage over 4,682,410 total running words and 82,145 unique vocabulary terms in the training split:

* **Token-level OOV Rate:** 1.24% (58,061 running words dropped)
* **Type-level OOV Rate:** 14.82% (12,174 distinct dictionary words missed)
* **Most Frequent OOV Words:** `flawless` (1,428), `underrated` (1,185), `crap` (962), `overrated` (894), `unwatchable` (672), `shyamalan` (580), `cheesy` (544), `laughable` (521).

Static word lookup tables discard sentiment-rich vocabulary like `unwatchable` and `underrated`. In contrast, subword tokenizers (WordPiece) split unseen terms into known subword units without discarding tokens.

---

## Qualitative Live Inference: Edge-Case Evaluation

Both Fine-Tuned DistilBERT and the TF-IDF baseline were evaluated on challenging unseen test cases covering clear sentiment, mixed reviews, complex negation, and sarcasm:

| Category                     | Full Review Text                                                                                                                            |    DistilBERT Prediction    |          TF-IDF Baseline          |    Ground Truth    | Analysis                                                                                                                                 |
| :--------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------ | :-------------------------: | :--------------------------------: | :----------------: | :--------------------------------------------------------------------------------------------------------------------------------------- |
| **Clear Positive**     | *"An absolute cinematic masterpiece! The performances were breathtaking and the storytelling was deeply moving from start to finish."*    |      Positive (98.69%)      |         Positive (77.09%)         |      Positive      | Both models correctly detect strong positive polarity.                                                                                   |
| **Clear Negative**     | *"A colossal waste of time and money. Painfully dull dialogue, atrocious acting, and a completely incoherent plot."*                      |      Negative (98.62%)      |         Negative (99.33%)         |      Negative      | Both models identify harsh negative criticism.                                                                                           |
| **Mixed / Nuanced**    | *"While the visual effects and cinematography were undeniably stunning, the sluggish pacing and hollow characters left me disappointed."* |      Negative (97.17%)      |         Negative (58.98%)         |      Negative      | DistilBERT weighs the concluding clause; TF-IDF is conflicted by positive terms (*stunning*).                                          |
| **Complex Negation**   | *"I walked in expecting a complete disaster, but it was not bad at all and actually quite charming and enjoyable."*                       | **Positive (97.63%)** | **Negative (63.33%) [FAIL]** | **Positive** | **Key finding:** TF-IDF fails due to negative word counts (*disaster*, *bad*). DistilBERT parses *not bad at all* correctly. |
| **Subtle / Sarcastic** | *"Oh brilliant, another completely predictable sequel filled with cliche tropes that nobody ever asked for."*                             |      Negative (98.18%)      |         Negative (68.59%)         |      Negative      | Both models detect negative sentiment through contextual clues (*predictable*, *cliche*).                                            |

---

## Deployment and Verification

The pipeline provides two validated serving options:

1. **High-Accuracy Deep Learning Tier (Fine-Tuned DistilBERT):**
   * Loaded via `DistilBertForSequenceClassification.from_pretrained('./saved_models/distilbert_sentiment')`.
   * Verified on unseen review: *"An extraordinary cinematic achievement with masterful acting and direction."*
   * Output: `Positive (Confidence: 98.36%)`.
2. **High-Throughput CPU Fallback Tier (TF-IDF + Logistic Regression):**
   * Loaded via `joblib.load('./saved_models/tfidf_pipeline.joblib')`.
   * Sub-millisecond execution time with zero GPU dependencies.
   * Output: `Positive (Confidence: 63.14%)`.

---

## Author

* **Md. Tausif Jafar**
* **Email :** mdtausifjafar@gmail.com
