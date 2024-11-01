import pickle
from flask import Flask, request, render_template
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load NLTK dependencies
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Creating the Flask app object
app = Flask(__name__)

try:
    # Loading the models
    count_vector = pickle.load(open('count_vector.pkl', 'rb'))
    feature_names = pickle.load(open('feature_names.pkl', 'rb'))
    tfidf_transformer = pickle.load(open('tfidf_transformer.pkl', 'rb'))
    
    # Ensure feature_names is a non-empty list
    if feature_names is None or not isinstance(feature_names, list) or len(feature_names) == 0:
        raise ValueError("Feature names are either missing or not loaded correctly. Check the model files.")

except FileNotFoundError as e:
    print("Model file not found:", e)
except ValueError as e:
    print("Model data error:", e)
except Exception as e:
    print("An unexpected error occurred:", e)

# Define stopwords
stop_words = set(stopwords.words('english'))
new_words = ["fig", "figure", "image", "sample", "using", "show", "result",
             "large", "also", "one", "two", "three", "four", "five", "seven", "eight", "nine"]
stop_words = list(stop_words.union(new_words))

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

def get_keywords(docs, topN):
    # Transform document to get word counts
    docs_word_count = tfidf_transformer.transform(count_vector.transform([docs]))  # Pass `docs` in a list
    docs_word_count = docs_word_count.tocoo()  # Convert to COO sparse matrix
    tuples = zip(docs_word_count.col, docs_word_count.data)
    sorted_items = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)[:topN]

    # Extract top N keywords
    results = {feature_names[idx]: round(score, 3) for idx, score in sorted_items}
    return results

# Custom text preprocessing function
def preprocess_text(text):
    if isinstance(text, float): 
        text = str(text)
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = nltk.word_tokenize(text)
    text = [PorterStemmer().stem(word) for word in text if word not in stop_words and len(word) >= 3]
    return ' '.join(text)

@app.route('/extract_keywords', methods=['POST', 'GET'])
def extract_keywords():
    file = request.files.get('file')
    if not file or file.filename == '':
        return render_template('index.html', error="No file selected")

    if file.content_type == 'text/plain':
        text = file.read().decode('utf-8', errors='ignore')
        cleaned_text = preprocess_text(text)
        keywords = get_keywords(cleaned_text, 20)
        return render_template('keywords.html', keywords=keywords)
    else:
        return render_template('index.html')

@app.route('/search_keywords', methods=['POST', 'GET'])
def search_keywords():
    search_query = request.form.get('search')
    if search_query:
        keywords = feature_names[:20]  # Limit to top 20 feature names
        print(keywords)
        return render_template('keywords.html', keywords=keywords)
    else:
        return render_template('index.html')

# Run the Flask application
if __name__ == "__main__":
    app.run(debug=True)
