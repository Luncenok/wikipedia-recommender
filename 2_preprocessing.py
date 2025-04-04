import spacy
import pandas as pd
import json

# Load SpaCy's English model
nlp = spacy.load("en_core_web_sm")

def preprocess_articles(articles):
    processed_articles = []
    i=0
    for article in articles:
        print(f"Preprocessing article {i}/1000")
        doc = nlp(article['content'])
        lemmatized = " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])
        processed_articles.append({"title": article['title'], "content": lemmatized})
        i+=1

    return processed_articles

# Load scraped articles
with open("wikipedia_articles.json", "r") as file:
    raw_articles = json.load(file)

# Preprocess articles
preprocessed_articles = preprocess_articles(raw_articles)

# Save to CSV
df = pd.DataFrame(preprocessed_articles)
df.to_csv("preprocessed_articles.csv", index=False)

# Save to JSON for embeddings
with open("preprocessed_articles.json", "w") as file:
    json.dump(preprocessed_articles, file)
