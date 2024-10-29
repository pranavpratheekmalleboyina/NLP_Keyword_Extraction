import pickle
from flask import Flask,request,render_template
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#Creating the Flask app object
app = Flask(__name__)

#loading the models
count_vector = pickle.load(open('count_vector.pkl','rb'))
feature_names = pickle.load(open('feature_names.pkl','rb'))
tfidfTransformer = pickle.load(open('tfidfTransformer.pkl','rb'))

stop_words = set(stopwords.words('english'))
new_words = ["fig","figure","image","sample","using","show","result",
             "large","also","one","two","three","four","five","seven","eight","nine"]
stop_words = list(stop_words.union(new_words))

#let us define the routes
@app.route('/')
def index():
    return render_template('index.html')

def get_keywords(docs,topN):
  docs_word_count = tfidfTransformer.transform(count_vector.transform([docs[idx]]))
  #build sparse matrix
  docs_word_count = docs_word_count.tocoo()
  tuples = zip(docs_word_count.col,docs_word_count.data)
  sorted_items = sorted(tuples,key = lambda x: (x[1],x[0]),reverse=True)

  sorted_items = sorted_items[:topN]
  score_vals = []
  feature_vals = []
  for idx, score in sorted_items:
    score_vals.append(round(score,3))
    feature_vals.append(feature_names[idx])

  results = {}
  for idx in range(len(feature_vals)):
    results[feature_vals[idx]] = score_vals[idx]
  return results

#custom functions
def preprocess_text(text):
  if isinstance(text, float): 
    text = str(text)
  text = text.lower()
  text = re.sub(r'<.*?>',' ',text)
  text = re.sub(r'[^a-zA-Z]',' ',text)
  text = nltk.word_tokenize(text)
  text = [word for word in text if word not in stop_words]
  text = [word for word in text if len(word) >= 3]
  stemming = PorterStemmer()
  text = [stemming.stem(word) for word in text]
  return ' '.join(text)

@app.route('/extract_keywords',methods=['POST','GET'])
def extract_keywords():
    file = request.file['file']
    if file.filename == '':
        return render_template('index.html',error="No file selected")
    if file:
        file.read().decode('utf-8',errors='ignore')      
        cleaned_file = preprocess_text(file) 
        keywords = get_keywords(cleaned_file,20)
    return render_template('keywords.html',keywords=keywords)    

@app.route('/search_keywords',methods=['POST','GET'])
def search_keywords():
    pass

#main function
if __name__ == "__main__":
    app.run(debug=True)
