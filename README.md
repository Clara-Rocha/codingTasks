# Sentiment Analysis with SpaCy

## Overview
This project is a Python script for sentiment analysis using SpaCy, a natural language processing library. It analyzes product reviews from the Consumer Reviews of Amazon Products dataset and predicts their sentiment (positive, negative, or neutral).

## Features
- Load the `en_core_web_sm` SpaCy model for natural language processing tasks.
- Preprocess text data by removing stopwords and performing basic text cleaning.
- Implement a sentiment analysis model using SpaCy.
- Evaluate the model's accuracy on sample product reviews.
- Generate a brief report summarizing the dataset, preprocessing steps, evaluation results, and insights into the model's strengths and limitations.

## Requirements
- Python 3.x
- SpaCy
- Pandas
- TextBlob
- FPDF

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/Clara-Rocha/sentiment-analysis.git
    ```

2. Navigate to the project directory:
    ```bash
    cd sentiment-analysis
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the SpaCy language model:
    ```bash
    python -m spacy download en_core_web_sm
    ```

## Usage
1. Place your `amazon_product_reviews.csv` file in the project directory.

2. Run the script:
    ```bash
    python sentiment_analysis.py
    ```

3. View the generated `sentiment_analysis_report.pdf` for a summary of the analysis.

## Contributing
Contributions are welcome! If you find any bugs or have suggestions for improvement, please open an issue or submit a pull request.
