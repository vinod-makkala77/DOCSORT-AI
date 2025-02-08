import fitz  # PyMuPDF
import os
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib

# Download stopwords (only once)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

def clean_text(text):
    """Clean extracted text: remove numbers, special characters, and stopwords."""
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Load trained model & vectorizer
model = joblib.load("document_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Folder containing PDFs
pdf_folder = "C:\\Users\\Vinod Makkala\\OneDrive\\Desktop\\many"

# Important keyword list
important_keywords = {"policy", "report", "urgent", "confidential", "meeting"}

# Files that should always be important
important_files = {"meeting_minutes.pdf", "financial_report.pdf"}

# Minimum document length for importance
length_threshold = 300  

# Minimum probability threshold for importance
probability_threshold = 0.3

# Process all PDFs in the folder
for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):  
        file_path = os.path.join(pdf_folder, filename)

        # Extract & clean text
        pdf_text = extract_text_from_pdf(file_path)
        pdf_text_cleaned = clean_text(pdf_text)

        # Transform text into features & classify
        X_input = vectorizer.transform([pdf_text_cleaned])
        prediction = model.predict(X_input)[0]

        # Document length check
        doc_length = len(pdf_text_cleaned.split())

        # Keyword check
        found_keywords = [word for word in pdf_text_cleaned.split() if word in important_keywords]

        # 🔹 Final decision logic
        if (
            prediction == 1 or 
            doc_length > length_threshold or 
            found_keywords or 
            filename in important_files
        ):
            label = "Important"
            print(f"✅ {filename} --> Important")
        else:
            label = "Unimportant"
            print(f"❌ {filename} --> Unimportant (Deleting...)")
            os.remove(file_path)  # 🚨 Delete unimportant PDF permanently

print("🔹 Processing Complete. Unimportant PDFs are deleted.")  
