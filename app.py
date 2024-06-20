from flask import Flask, render_template_string, request
import pandas as pd
import replicate
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)

# Set your Replicate API token
api_token = 'r8_APt6Od6nyVTF9btCWVPaapsv61sHBc23TAK2P'
os.environ['REPLICATE_API_TOKEN'] = api_token

# Load the dataset and prepare the articles
news_articles = pd.read_json('News_Category_Dataset_v3.json', lines=True)

# Filter and preprocess data
news_articles = news_articles[news_articles['date'] >= pd.Timestamp(2018, 1, 1)]
news_articles = news_articles[news_articles['headline'].apply(lambda x: len(x.split()) > 5)]
news_articles.sort_values('headline', inplace=True, ascending=False)
duplicated_articles_series = news_articles.duplicated('headline', keep=False)
news_articles = news_articles[~duplicated_articles_series]
news_articles.index = range(news_articles.shape[0])

# Preprocess headlines for TF-IDF
stop_words = set(stopwords.words('english'))
news_articles_temp = news_articles.copy()

for i in range(len(news_articles_temp["headline"])):
    string = ""
    for word in news_articles_temp["headline"][i].split():
        word = ("".join(e for e in word if e.isalnum()))
        word = word.lower()
        if word not in stop_words:
            string += word + " "
    news_articles_temp.at[i, "headline"] = string.strip()

# Create TF-IDF matrix
tfidf_headline_vectorizer = TfidfVectorizer(min_df=0.0)
tfidf_headline_features = tfidf_headline_vectorizer.fit_transform(news_articles_temp['headline'])

# Function to find similar articles based on the headline
def tfidf_based_model(row_index, num_similar_items):
    couple_dist = pairwise_distances(tfidf_headline_features, tfidf_headline_features[row_index])
    indices = np.argsort(couple_dist.ravel())[0:num_similar_items + 1]
    return news_articles.iloc[indices[1:],]  # Exclude the first one since it's the queried article itself

# HTML template for the main page
template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>E Team Gazette</title>
    <link rel="icon" href="/static/n_Icon.jpg" type="image/x-icon">
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
            background-color: #f4f4f4;
            color: #333;
        }
        .container {
            max-width: 800px;
            margin: auto;
            padding: 20px;
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        header {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 20px;
        }
        header img {
            height: 60px;  /* Increased size of the icon */
            width: 60px;   /* Maintain aspect ratio with increased size */
            margin-right: 15px;  /* Increased spacing between icon and title */
        }
        header h1 {
            font-size: 2.5em; /* Increased font size of the title */
            color: #2c3e50;
            margin: 0;
        }
        .article {
            margin-bottom: 30px;
            padding: 10px;
            border-bottom: 1px solid #ccc;
        }
        .article h2 {
            margin-top: 0;
            color: #3498db;
        }
        .article .category {
            font-weight: bold;
            color: #e74c3c;
        }
        .article .date {
            font-style: italic;
            color: #888;
        }
    </style>
</head>
<body>
    <header>
        <img src="/static/n_Icon.jpg" alt="Icon">
        <h1>E Team Gazette</h1>
    </header>
    <div class="container">
        {% for article in articles %}
        <div class="article">
            <h2>{{ article.headline }}</h2>
            <p class="category">{{ article.category }}</p>
            <p>{{ article.short_description }}</p>
            <p><strong>Author(s):</strong> {{ article.authors }}</p>
            <p class="date">{{ article.date.strftime('%B %d, %Y') }}</p>
            <p><a href="/generate-text?headline={{ article.headline|urlencode }}&short_description={{ article.short_description|urlencode }}&authors={{ article.authors|urlencode }}&category={{ article.category|urlencode }}&date={{ article.date|urlencode }}" target="_blank">Read more</a></p>
        </div>
        {% endfor %}
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    # Get the latest 6 articles
    latest_articles = news_articles.head(6)
    articles_list = latest_articles.to_dict(orient='records')
    return render_template_string(template, articles=articles_list)

@app.route('/generate-text')
def generate_text():
    # Retrieve query parameters
    headline = request.args.get('headline')
    short_description = request.args.get('short_description')
    authors = request.args.get('authors')
    category = request.args.get('category')
    date = request.args.get('date')

    # Generate extended content using Replicate API
    generated_text = ""
    for event in replicate.stream(
        "meta/meta-llama-3-70b-instruct",
        input={"prompt": short_description}
    ):
        generated_text += str(event)

    # Find the index of the current article
    article_index = news_articles[news_articles['headline'] == headline].index[0]

    # Get similar articles
    similar_articles = tfidf_based_model(article_index, 4)
    similar_articles_list = similar_articles.to_dict(orient='records')

    # Render a new page with the generated content and article details
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{{ headline }}</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 20px;
                background-color: #f4f4f4;
                color: #333;
            }
            .container {
                max-width: 800px;
                margin: auto;
                padding: 20px;
                background: #fff;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }
            h1 {
                text-align: center;
                color: #2c3e50;
            }
            p {
                text-align: justify;
            }
            .details {
                margin-top: 20px;
                font-size: 1.1em;
                color: #555;
            }
            .recommendations {
                margin-top: 40px;
            }
            .recommendations h2 {
                margin-top: 0;
                color: #3498db;
            }
            .recommendations .article {
                margin-bottom: 20px;
                padding: 10px;
                border-bottom: 1px solid #ccc;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{{ headline }}</h1>
            <p class="details"><strong>Category:</strong> {{ category }}</p>
            <p class="details"><strong>Author(s):</strong> {{ authors }}</p>
            <p class="details"><strong>Date:</strong> {{ date }}</p>
            <hr>
            <p>{{ extended_text }}</p>
            <hr>
            <div class="recommendations">
                <h2>Recommended Articles</h2>
                {% for article in recommendations %}
                <div class="article">
                    <h3>{{ article.headline }}</h3>
                    <p class="category">{{ article.category }}</p>
                    <p>{{ article.short_description }}</p>
                    <p><strong>Author(s):</strong> {{ article.authors }}</p>
                    <p class="date">{{ article.date.strftime('%B %d, %Y') }}</p>
                </div>
                {% endfor %}
            </div>
            <p><a href="/">Back to Home</a></p>
        </div>
    </body>
    </html>
    """, headline=headline, extended_text=generated_text, authors=authors, category=category, date=date, recommendations=similar_articles_list)

if __name__ == '__main__':
    app.run(debug=True)
