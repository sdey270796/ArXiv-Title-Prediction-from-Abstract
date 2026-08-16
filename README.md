# ArXiv Title Generation from Abstracts

> **NLP / Text Generation Project using BART**

This project explores automated **academic paper title generation from
research abstracts** using a Transformer-based sequence-to-sequence
model. The notebook uses metadata from the **arXiv** repository and
fine-tunes a pre-trained **BART-base** model to generate a paper title
from its abstract.

The project also evaluates the generated titles using **ROUGE-L, BLEU,
and exact-match accuracy**, followed by a detailed error-analysis and
visualization stage.

------------------------------------------------------------------------

## 📌 Project Overview

With the rapid growth of academic publications, writing concise and
informative titles can be time-consuming. This project investigates
whether a neural text-generation model can learn the relationship
between an academic paper's abstract and its corresponding title.

The task is formulated as:

**Input:** Research paper abstract\
**Output:** Predicted research paper title

The implementation uses the `facebook/bart-base` Transformer model
through the `Simple Transformers` sequence-to-sequence framework.

------------------------------------------------------------------------

## 🎯 Objectives

-   Automate academic title generation from paper abstracts.
-   Fine-tune a pre-trained Transformer model for the title-generation
    task.
-   Evaluate generated titles against their reference titles.
-   Measure both lexical overlap and exact matching.
-   Analyze the distribution and relationship of ROUGE and BLEU scores.
-   Identify areas where the title-generation system can be improved.

------------------------------------------------------------------------

## 🗂️ Dataset

The project uses the **arXiv metadata snapshot**:

`arxiv-metadata-oai-snapshot.json`

The notebook reads the large JSON metadata file line-by-line using a
generator rather than loading the complete JSON file into memory at
once.

The relevant metadata fields are:

-   `title`
-   `abstract`
-   `categories`
-   `journal-ref`

### Dataset Filtering

The notebook:

1.  Reads the arXiv metadata snapshot.
2.  Uses the category mapping defined in the notebook.
3.  Considers the categories represented in that mapping.
4.  Extracts papers whose journal reference year satisfies:

`2010 < year < 2021`

5.  Extracts the paper title and abstract.
6.  Removes newline characters from abstracts.
7.  Removes missing records.
8.  Renames the final columns to:
    -   `input_text` → abstract
    -   `target_text` → title

After preprocessing, the notebook reports:

**21,641 paper records with 2 columns.**

------------------------------------------------------------------------

## 🧠 Problem Formulation

The project treats title generation as a **sequence-to-sequence
text-generation problem**.

``` text
Research Abstract
       │
       ▼
BART Encoder
       │
       ▼
Transformer Representation
       │
       ▼
BART Decoder
       │
       ▼
Generated Research Title
```

Unlike a conventional classification problem, the model is required to
generate a new sequence of words. Consequently, exact string matching is
a particularly strict evaluation criterion.

------------------------------------------------------------------------

## 🤖 Model

### BART

The notebook uses:

**Model:** `facebook/bart-base`

BART is a Transformer-based encoder-decoder architecture designed for
sequence-to-sequence language tasks, including text generation and
summarization.

For this project:

``` text
Abstract → BART → Title
```

The model is initialized through:

``` python
Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name="facebook/bart-base",
    args=model_args
)
```

### Training Configuration

The notebook uses the following configuration:

  Parameter                                  Value
  ------------------------- ----------------------
  Base model                  `facebook/bart-base`
  Maximum sequence length                      512
  Training batch size                            6
  Training epochs                                3
  Evaluation fraction                          10%
  Random state                                  42
  Training framework           Simple Transformers
  Deep learning framework                  PyTorch

------------------------------------------------------------------------

## 🔀 Train / Evaluation Split

The notebook randomly samples **10%** of the processed dataset for
evaluation:

``` python
eval_df = papers.sample(frac=0.1, random_state=42)
train_df = papers.drop(eval_df.index)
```

With 21,641 records, this corresponds to approximately:

-   **Training:** 19,477 papers
-   **Evaluation:** 2,164 papers

The evaluation split is fixed using `random_state=42`.

------------------------------------------------------------------------

## 🔄 Project Workflow

``` text
                    ┌─────────────────────────┐
                    │ arXiv Metadata Snapshot │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Metadata Extraction     │
                    │ & Year Filtering        │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Abstract / Title        │
                    │ Dataset Construction    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Train / Evaluation Split│
                    │ 90% / 10%               │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ BART-base Fine-tuning   │
                    │ 3 Epochs                │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Title Generation        │
                    └────────────┬────────────┘
                                 │
                                 ▼
             ┌───────────────────┼───────────────────┐
             ▼                   ▼                   ▼
          ROUGE-L              BLEU            Exact Match
             │                   │                   │
             └───────────────────┼───────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │ Error Analysis &        │
                    │ Visualization           │
                    └─────────────────────────┘
```

------------------------------------------------------------------------

## 📊 Evaluation Metrics

Three metrics are calculated for generated titles.

### 1. ROUGE-L

ROUGE-L evaluates the overlap between the generated title and the
reference title using the **Longest Common Subsequence (LCS)**.

A higher score indicates greater sequence-level similarity.

The notebook reports an average:

**ROUGE-L = 0.369**

------------------------------------------------------------------------

### 2. BLEU

BLEU evaluates the overlap of word n-grams between the generated title
and the reference title.

The notebook reports an average:

**BLEU = 0.354**

------------------------------------------------------------------------

### 3. Exact-Match Accuracy

A prediction receives a value of `1` only when:

``` text
Predicted Title == True Title
```

Otherwise it receives `0`.

The notebook reports:

**Exact-Match Accuracy = 0.004 (0.4%)**

This is much stricter than ROUGE or BLEU because semantically similar
titles can still have completely different wording.

------------------------------------------------------------------------

## 📈 Model Results

  Metric                       Result
  ---------------------- ------------
  Average ROUGE-L           **0.369**
  Average BLEU              **0.354**
  Exact-Match Accuracy       **0.4%**
  Evaluation loss          **2.2129**

### Interpretation

The results indicate that the BART model is able to generate titles with
a **moderate degree of lexical and sequence-level similarity** to the
reference titles.

However, exact reproduction of the original title is rare.

This distinction is important for title generation: multiple different
titles can plausibly describe the same abstract, so exact-match accuracy
is a very restrictive measure for this task.

------------------------------------------------------------------------

## 🧪 Prediction Analysis

The notebook generates predictions for **250 randomly selected samples**
from the evaluation dataset.

For each sampled paper, the project records:

-   Abstract
-   True title
-   Predicted title
-   ROUGE-L score
-   BLEU score
-   Exact-match indicator

The predictions are also written to:

``` text
random_sample_predictions_with_metrics.txt
```

A structured DataFrame is created with:

``` text
Abstract
True Title
Predicted Title
ROUGE Score
BLEU Score
Accuracy
```

### Example

One of the notebook's examples shows:

**True title**

> Sieve-SDP: a simple facial reduction algorithm...

**Generated title**

> Sieve-SDP: A Simple Facial Reduction Algorithm...

This example illustrates a useful property of the model: a generated
title can be highly similar to the reference while still receiving zero
exact-match accuracy because of differences in wording or
capitalization.

------------------------------------------------------------------------

## 📊 Exploratory and Error Analysis

The notebook contains a substantial post-training analysis of the
generated titles.

### ROUGE Distribution

The distribution of ROUGE-L scores is examined to understand how
frequently the model produces titles with low, moderate, or high overlap
with the reference titles.

### BLEU Distribution

BLEU scores are similarly analyzed to understand the distribution of
n-gram overlap.

### Exact-Match Distribution

The notebook examines the distribution of exact-match results,
illustrating how uncommon perfect title reproduction is.

### ROUGE vs BLEU

A scatter plot compares ROUGE-L and BLEU scores for individual
predictions.

This helps identify:

-   Predictions that perform well on both metrics.
-   Predictions with high ROUGE but lower BLEU.
-   Predictions with high BLEU but lower ROUGE.
-   Poor predictions where both metrics are low.

### Box Plots

Box plots are used to examine:

-   Median performance.
-   Interquartile ranges.
-   Score variability.
-   Potential outliers.

### Swarm Plots

Swarm plots provide a point-level view of the ROUGE and BLEU
distributions.

### Pair Plot

A pair plot is used to visualize relationships between ROUGE and BLEU
scores.

### Violin Plots

Violin plots provide an additional view of the distribution and density
of ROUGE and BLEU scores.

Together, these analyses provide a broader picture of model behavior
rather than relying only on average evaluation metrics.

------------------------------------------------------------------------

## 🛠️ Technologies Used

### Programming Language

-   Python

### Machine Learning / Deep Learning

-   PyTorch
-   Hugging Face Transformers
-   Simple Transformers

### NLP

-   BART
-   Sequence-to-sequence learning
-   Text generation

### Data Processing

-   NumPy
-   Pandas
-   JSON

### Evaluation

-   ROUGE
-   BLEU
-   NLTK

### Visualization

-   Matplotlib
-   Seaborn
-   Plotly

### Utilities

-   tqdm
-   logging

------------------------------------------------------------------------

## 📁 Repository Structure

The exact repository structure can be adapted to the files stored
alongside this notebook. A typical structure for this project is:

``` text
.
├── README.md
├── arxiv_title_prediction_from_abstract.ipynb
├── arxiv-metadata-oai-snapshot.json
├── random_sample_predictions_with_metrics.txt
└── ...
```

The notebook itself contains the complete data-processing, training,
prediction, evaluation, and analysis workflow.

------------------------------------------------------------------------

## 🚀 How to Run

### 1. Clone the repository

``` bash
git clone <your-repository-url>
cd <repository-name>
```

### 2. Install the required packages

The notebook installs/configures the packages it requires. The main
ecosystem includes:

``` bash
pip install numpy pandas matplotlib seaborn plotly
pip install torch transformers tokenizers
pip install simpletransformers
pip install nltk rouge-score
```

> The exact package versions should follow the environment used by the
> notebook.

### 3. Place the Dataset

Ensure that the arXiv metadata snapshot is available at the path
expected by the notebook:

``` text
../input/arxiv/arxiv-metadata-oai-snapshot.json
```

If running in Google Colab, the notebook also contains an alternative
dataset-loading approach.

### 4. Open the Notebook

Run:

``` text
arxiv_title_prediction_from_abstract.ipynb
```

The notebook performs:

1.  Dataset loading
2.  Metadata inspection
3.  Category mapping
4.  Filtering
5.  Dataset construction
6.  Train/evaluation splitting
7.  BART initialization
8.  Model training
9.  Evaluation
10. Title generation
11. ROUGE/BLEU/exact-match calculation
12. Error analysis
13. Visualization

------------------------------------------------------------------------

## 💡 Key Findings

The experiment demonstrates several important points:

1.  **BART can learn the abstract-to-title mapping sufficiently well to
    generate titles with moderate lexical similarity.**

2.  **ROUGE-L and BLEU provide substantially more useful information
    than exact-match accuracy for this generation task.**

3.  **Exact title reproduction is difficult**, with only 0.4% of the
    sampled predictions matching the reference titles exactly.

4.  **Generated titles can be highly similar without being exact
    matches.** This is visible in examples where the predicted title
    captures the subject and wording of the reference title but differs
    slightly in phrasing.

5.  **Model performance varies considerably across individual papers**,
    motivating the detailed error-analysis stage included in the
    notebook.

------------------------------------------------------------------------

## ⚠️ Limitations

The notebook represents a useful experimental implementation, but
several limitations should be considered.

### Limited Training Configuration

The model is trained for only **3 epochs** with a batch size of **6**.
More extensive hyperparameter tuning may improve performance.

### Exact Match Is Too Strict

A generated title can be relevant and semantically correct while
differing from the original title. Therefore, exact-match accuracy
should not be interpreted as the sole measure of title-generation
quality.

### Evaluation Sampling

The detailed prediction analysis uses **250 randomly selected draws**
from the evaluation dataset. Because sampling is random and performed
with replacement, these 250 predictions are an analysis sample rather
than the complete 2,164-paper evaluation set.

### Domain Diversity

The dataset spans multiple arXiv research categories. Different
scientific domains have very different title-writing conventions, which
can make a single model's performance heterogeneous.

### Metric Limitations

ROUGE and BLEU primarily measure lexical overlap. They do not fully
capture whether a generated title is scientifically meaningful,
factually appropriate, or semantically equivalent to the reference
title.

------------------------------------------------------------------------

## 🔮 Future Improvements

The notebook itself suggests several directions for further development:

-   Perform deeper error analysis on consistently poor predictions.
-   Fine-tune the model with additional or domain-specific data.
-   Experiment with alternative Transformer architectures.
-   Explore beam-search and other decoding strategies.
-   Compare BART against models such as T5 and other encoder-decoder
    architectures.
-   Evaluate semantic similarity using embedding-based metrics.
-   Investigate domain-specific models for scientific text.
-   Perform more systematic hyperparameter tuning.
-   Evaluate the complete held-out test set rather than relying
    primarily on a sampled prediction analysis.
-   Introduce human evaluation to assess relevance, informativeness, and
    readability of generated titles.

------------------------------------------------------------------------

## 📚 Conceptual Takeaway

This project demonstrates how a **pre-trained Transformer can be adapted
to a specialized scientific NLP generation problem**.

The central idea is simple:

``` text
                    ABSTRACT
                       │
                       ▼
              ┌─────────────────┐
              │   BART Encoder  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ BART Transformer│
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   BART Decoder  │
              └────────┬────────┘
                       │
                       ▼
                 GENERATED TITLE
```

The experiment shows that the model can capture meaningful relationships
between abstracts and titles, while also highlighting the difficulty of
evaluating open-ended text generation using exact string matching.

------------------------------------------------------------------------

## 👤 Author

**Subhadeep Dey**

This project is part of a broader exploration of **Natural Language
Processing, Generative AI, Machine Learning, and scientific text
analysis**.

------------------------------------------------------------------------

## ⭐ If You Find This Project Useful

Consider starring the repository and exploring the notebook for the
complete implementation, experiments, visualizations, and generated
predictions.
