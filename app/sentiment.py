from flask import Blueprint, render_template, request
from flask_login import login_required
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sentiment = Blueprint('sentiment', __name__, url_prefix='/sentiment')

# load once
analyzer = SentimentIntensityAnalyzer()

@sentiment.route('/', methods=['GET', 'POST'])
@login_required
def analyze():
    result = None
    if request.method == 'POST':
        txt = request.form.get('text', '').strip()
        if txt:
            scores = analyzer.polarity_scores(txt)
            result = {
                'neg': round(scores['neg'], 3),
                'neu': round(scores['neu'], 3),
                'pos': round(scores['pos'], 3),
            }
    return render_template('sentiment.html', result=result)