from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os

app = Flask(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'spam_model.pkl')
vectorizer_path = os.path.join(current_dir, 'tfidf_vectorizer.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)
with open(vectorizer_path, 'rb') as f:
    tfidf = pickle.load(f)

@app.route('/')
def home():
    return render_template('home.html', div3_vis="hidden")

@app.route('/check', methods=['POST'])
def check_spam():
    email_content = request.form.get('em')
    
    email_tfidf = tfidf.transform([email_content])
    
    prediction = model.predict(email_tfidf)[0]
    
    if prediction == 1:
        return render_template('home.html',div3_vis='visible', is_spam=True) 
    else:
        return render_template('home.html',div3_vis='visible',is_spam=False)

if __name__ == "__main__":
    app.run(debug=True)