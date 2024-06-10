import spacy
import pandas as pd
from textblob import TextBlob
from fpdf import FPDF

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load the dataset
dataframe = pd.read_csv('C:/Users/croch/OneDrive/Desktop/CoGrammar/Tasks/lasttask/amazon_products_review.csv', low_memory=False) # columns 1&10 with problems

# Select the 'review.text' column and remove missing values
reviews_data = dataframe['reviews.text']
clean_data = reviews_data.dropna()

# Function to preprocess text
def preprocess_text(text):
    """
    Preprocess the input text by converting it to lowercase, stripping
    whitespace, tokenizing, lemmatizing, and removing stopwords.
    """
    # Convert to lowercase and strip whitespace
    text = text.lower().strip()
    doc = nlp(text)
    # Filter out stop words and non-alphabetic tokens
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(tokens)

# Apply preprocessing to the reviews
clean_data = clean_data.apply(preprocess_text)

# Function to perform sentiment analysis using TextBlob
def analyze_sentiment(text):
    """
    Analyze the sentiment of the input text using TextBlob.
    Returns the polarity score ranging from -1 (negative) to 1 (positive).
    """
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Apply sentiment analysis to the preprocessed reviews
clean_data_sentiments = clean_data.apply(analyze_sentiment)

# Save the results to a new CSV file
sentiment_results = pd.DataFrame({
    'review': reviews_data,
    'cleaned_review': clean_data,
    'sentiment_score': clean_data_sentiments
})

sentiment_results.to_csv('sentiment_analysis_results.csv', index=False)

print("The sentiment analysis was completed and results were saved to sentiment_analysis_results.csv")

# Test the model on sample product reviews
sample_reviews = [
    "This product is amazing! I'm very happy with my purchase.",
    "I hated this product. It broke after one use.",
    "The quality is decent for the price, but I've seen better.",
    "Absolutely fantastic! Exceeded my expectations.",
    "Not worth the money. Very disappointed."
]

# Preprocess and analyze sentiment of sample reviews
for review in sample_reviews:
    cleaned_review = preprocess_text(review)
    sentiment_score = analyze_sentiment(cleaned_review)
    print(f"Review: {review}\nCleaned Review: {cleaned_review}\nSentiment Score: {sentiment_score}\n")

# Compare similarity of two product reviews
# Ensure the indices are within bounds
if len(clean_data) > 1:
    review_1 = clean_data.iloc[0]
    review_2 = clean_data.iloc[1]

    doc1 = nlp(review_1)
    doc2 = nlp(review_2)

    similarity_score = doc1.similarity(doc2)
    print(f"Similarity between review 1 and review 2: {similarity_score}")

# Generate the PDF report
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Sentiment Analysis Report', 0, 1, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(10)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

# Create PDF
pdf = PDFReport()

# Add a page
pdf.add_page()

# Add content
pdf.chapter_title('1. Description of the Dataset Used')
pdf.chapter_body('The dataset used is a collection of Amazon product reviews. It includes various reviews written by customers about different products.')

pdf.chapter_title('2. Details of the Preprocessing Steps')
pdf.chapter_body('The preprocessing steps involved converting text to lowercase, stripping whitespace, tokenizing using SpaCy, lemmatizing, and removing stopwords.')

pdf.chapter_title('3. Evaluation of Results')
pdf.chapter_body('The sentiment analysis was performed using TextBlob. Sample results include:\n'
                 '1. "This product is amazing! I\'m very happy with my purchase." -> Positive sentiment\n'
                 '2. "I hated this product. It broke after one use." -> Negative sentiment\n'
                 '3. "The quality is decent for the price, but I\'ve seen better." -> Neutral sentiment\n'
                 '4. "Absolutely fantastic! Exceeded my expectations." -> Positive sentiment\n'
                 '5. "Not worth the money. Very disappointed." -> Negative sentiment')

pdf.chapter_title('4. Insights into the Model\'s Strengths and Limitations')
pdf.chapter_body('Strengths: Simple and quick to implement, effective for basic sentiment analysis.\n'
                 'Limitations: May not handle complex sentences well, limited by the capabilities of the TextBlob library.\n'
                 'Similarity comparison: Two reviews were compared using SpaCy\'s similarity function, yielding a similarity score of: {:.2f}'.format(similarity_score))

# Save the PDF
pdf.output('sentiment_analysis_report.pdf')

print("Report generated and saved as sentiment_analysis_report.pdf")