import os
import spacy
import pandas as pd
import json
import matplotlib.pyplot as plt
from newsapi import NewsApiClient
from datetime import datetime, timedelta
from string import punctuation
from collections import Counter
from wordcloud import WordCloud

pos_tag = ['VERB', 'NOUN', "PROPN"]
articles = []
dados = []

# uses environment variable to access api key
spacy_key = os.getenv('SPACY_KEY')
nlp_eng = spacy.load('en_core_web_lg')
newsapi = NewsApiClient (api_key=spacy_key)

start_date = datetime.now() - timedelta(days=28)
end_date = datetime.now()

# pagination
for page_num in range(1,6):
    temp = newsapi.get_everything(q='coronavirus', language='en', from_param=start_date, to=end_date, sort_by='relevancy', page=page_num)
    articles.extend(temp["articles"])

# dump json
with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(articles, f, ensure_ascii=False, indent=4)

# serialize and clean data into a dataframe
dados = []
for i, article in enumerate(articles):
    for x in articles:
        title = x['title']
        description = x['description']
        content = x['content']
        dados.append({'title':title, 'desc':description, 'content':content})

df = pd.DataFrame(dados)
df = df.dropna()
df.head()

df.to_csv('data.csv',index=False)

# scraping content for keywords
def get_keywords_eng(text):
    result = []

    data = nlp_eng(text)
    for token in data:
        if (token.text in nlp_eng.Defaults.stop_words or token.text in punctuation):
            continue
        if (token.pos_ in pos_tag):
            result.append(token.text)
    return result

results = []
for content in df.values:
    results.append([('#' + x[0]) for x in Counter(get_keywords_eng(str(content))).most_common(5)])

df['keywords'] = results
df['keywords'].to_csv("keywords.csv", index=False)

text = str(results)
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()