import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import nltk
import re

try:
    from transformers import pipeline
except ImportError:
    import transformers
    pipeline = transformers.pipelines.pipeline

nltk.download('punkt')

st.set_page_config(page_title="Ghana News Buzz!", page_icon=":newspaper:")

# Function to scrape and summarize articles
@st.cache_data
def get_links(link):
    articles = {}
    r = requests.get(link)
    soup = BeautifulSoup(r.content, "html.parser")

    if r.status_code != 200:
        st.error("Error. Something is wrong here")
        return articles

    # Scraping news from MyJoyOnline
    if link == 'https://www.myjoyonline.com/':
        start = soup.findAll('div', class_='home-post-list-title')
        for news in start:
            a_tag = news.find('a')
            if a_tag:
                link = a_tag['href']
                news_text = a_tag.find('h4').text
                articles[news_text] = {"link": link}

                article_response = requests.get(link)
                article_soup = BeautifulSoup(article_response.content, 'html.parser')

                summary = article_soup.findAll('div', id="article-text", class_="mt-3 article-text")
                article_summary = ""
                for s in summary:
                    p_tags = s.find_all('p')
                    for p in p_tags:
                        article_summary += p.text.strip() + " "
                articles[news_text]["summary"] = article_summary.strip()

    # Scraping news from Pulse Ghana
    elif link == "https://www.pulse.com.gh/":
        for links in soup.findAll('a', attrs={'href': re.compile("^http")}):
            news_text = links.get('title')
            if news_text:
                link = links.get('href')
                articles[news_text] = {"link": link}

                article_response = requests.get(link)
                article_soup = BeautifulSoup(article_response.content, 'html.parser')

                summary = article_soup.findAll('div', id="lead", class_="article-perex")
                article_summary = ""
                for s in summary:
                    p_tags = s.find_all('p')
                    for p in p_tags:
                        article_summary += p.text.strip() + " "
                if article_summary:
                    articles[news_text]["summary"] = article_summary.strip()

    # Scraping from Yen news
    elif link == "https://yen.com.gh/ghana/":
        start = soup.findAll('article', class_='c-article-card')
        for links in start:
            a_tag = links.find('a')
            if a_tag:
                link = a_tag['href']
                news_text = a_tag.find('span').get_text()
                articles[news_text] = {"link": link}

                article_response = requests.get("https://yen.com.gh/ghana/")
                article_soup = BeautifulSoup(article_response.content, 'html.parser')

                summary_elements = article_soup.find_all('p', class_='align-justify')
                article_summary = ""
                for s in summary_elements:
                    article_summary += s.text.strip() + " "
                articles[news_text]['summary'] = article_summary.strip()

    # Scraping news from GBC Ghana
    elif link == "https://www.gbcghanaonline.com/":
        start = soup.findAll('div', class_='elementor-post__text')
        for links in start:
            a_tag = links.find('a')
            if a_tag:
                link = a_tag['href']
                news_text = a_tag.text.strip()
                articles[news_text] = {"link": link}

                article_response = requests.get(link)
                article_soup = BeautifulSoup(article_response.content, 'html.parser')

                sum_start = article_soup.findAll('div', class_='e-con-inner')
                summary = article_soup.findAll('div', class_='elementor-widget-container')
                article_summary = ""
                for s in summary:
                    p_tags = s.find_all('p')
                    for p in p_tags:
                        article_summary += p.text.strip() + " "
                articles[news_text] = {"link": link, "summary": article_summary.strip()}

    return articles

# Function to perform clustering, find the hottest topic, and analyze sentiment
@st.cache_data
def find_hottest_topic_and_sentiment(articles):
    if not articles:
        return None, None, None

    df = pd.DataFrame.from_dict(articles, orient='index')
    df['summary'] = df['summary'].fillna('')

    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    Y = vectorizer.fit_transform(df['summary'])

    dbs = DBSCAN(eps=0.7, min_samples=3, metric='cosine').fit(Y)
    df['Cluster'] = dbs.labels_

    # Exclude cluster -1 (noise) from the analysis
    filtered_df = df[df['Cluster'] != -1]

    if filtered_df['Cluster'].nunique() <= 1:
        return df, None, None

    sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

    # Perform sentiment analysis
    filtered_df['Sentiment'] = filtered_df['summary'].apply(lambda x: sentiment_pipeline(x[:512])[0]['label'])

    # Map sentiment labels to stars
    sentiment_map = {'1 star': 1, '2 stars': 2, '3 stars': 3, '4 stars': 4, '5 stars': 5}
    filtered_df['Stars'] = filtered_df['Sentiment'].apply(lambda x: sentiment_map[x])

    hottest_cluster = filtered_df['Cluster'].value_counts().idxmax()
    hottest_articles = filtered_df[filtered_df['Cluster'] == hottest_cluster]

    return df, hottest_cluster, hottest_articles

# Mapping sentiments to star ratings with emojis
def sentiment_to_stars(stars):
    return "⭐" * stars

# Streamlit App Layout
st.title("Hottest Topic in Town Finder")
st.markdown("### News in Ghana for Today")

# Predefined URLs
urls = [
    'https://www.myjoyonline.com/',
    'https://www.pulse.com.gh/',
    'https://yen.com.gh/ghana/',
    'https://www.gbcghanaonline.com/'
]

articles = {}
for url in urls:
    articles.update(get_links(url))

# Display the articles
col1, col2 = st.columns(2)
with col1:
    for title, info in articles.items():
        st.write(f"**Title:** {title}")
        st.write(f"**Link:** {info['link']}")
        st.write(f"**Summary:** {info.get('summary', 'No summary available.')}")
        st.write("---")

with col2:
    if st.button("Find Hottest Topic"):
        with st.spinner("Processing..."):
            df, hottest_cluster, hottest_articles = find_hottest_topic_and_sentiment(articles)

        if hottest_articles is not None:
            st.success("Hottest Topic Found!")
            for title, row in hottest_articles.iterrows():
                st.write(f"**Title:** {title}")
                st.write(f"**Link:** {row['link']}")
                st.write(f"**Summary:** {row['summary']}")
                st.write(f"**Sentiment:** {sentiment_to_stars(row['Stars'])}")
                st.write("---")
        else:
            st.error("No distinct hottest topic found. Please try with more websites or different URLs.")

# Adding some styling to make the app visually appealing
st.markdown(
    """
    <style>
    .reportview-container {
        background: #000000;
        color: #FFFFFF;
    }
    .sidebar .sidebar-content {
        background: #31333F;
        color: white;
    }
    h1, .stButton>button {
        color: #FFFFFF;
    }
    .stButton>button {
        background-color: #31333F;
        color: white;
    }
    .stSpinner {
        color: #FFFFFF;
    }
    .css-18e3th9 {
        background-color: #31333F;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)
