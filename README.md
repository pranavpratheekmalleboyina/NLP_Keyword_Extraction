# Keywords Extraction Project

This is a Flask based web application used for extracting keywords from a document.

## Features
- Extracting the most significant keywords from a document.
- Searching for relevant keywords in a document.

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/pranavpratheekmalleboyina/NLP_Keyword_Extraction.git
cd NLP_Keyword_Extraction
```

2. **Setup a virtual environment(optional)**
```bash
python3 -m venv env
source env/bin/activate
```

3. **Install the required dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the NLTK stopwords and punkt data**
```python
import nltk
nltk.download("punkt")
nltk.download("stopwords")
```

5.**Start the Flask server** 
```bash
python app.py
```
The server would run on `http://127.0.0.1:5000`

## Usage
1. Navigate to `http://127.0.0.1:5000` in your browser.
2. Upload a file in `.txt` format or search the keyword that you want to search.
3. View the result based on the action that you have performed.

## Folder Structure
- **app.py:** Main application file <br>
- **templates/:** HTML templates for rendering the web pages<br>
    - `home.html`: The homepage which contains the functionalities that we perform.
    - `extractedkeywords.html`:The page where the keywords extracted from the file are displayed along with their frequencies.
    - `searchedkeywordslist.html`:The page where the list of all words containing a particular keyword are displayed.
    - `emptykeywordslist.html`:The page that is displayed when a particular keyword does not occur in the file selected.
- **static:** Folder for static files such as Javascript and CSS. 
- **Model Files**
  
## Dependencies
- `Flask`: Web Framework for Python
- `NLTK`: Library for text processing(tokens,stopwords)
- `scikit-learn`: Provides CountVectorizer and TfIdfTransformer for keyword extraction.
- `pickle`: Used to load and save pre-trained models (`tfidf_transformer.pkl`, `count_vector.pkl` and `feature_names.pkl`).

## Training Data 
 The initial data was trained on text data from academic data. Preprocessing steps were applied to filter out custom stop words and transform text using stemming and tokenization.

## Model Files
The project uses the following pickle files , used for loading the pre-trained model components:<br>
- `feature_names.pkl`: contains a list of feature names obtained from the trained vocabulary.<br>
- `count_vector.pkl`: contains the word vectors for the vocabulary. <br>
- `tfidf_transformer.pkl`: for identifying the most significant keywords based on the tf-idf word scores.<br>

## Contribution
Open for improvements and new ideas. If interested to contribute to this project, please fork the repository and submit a pull request.
