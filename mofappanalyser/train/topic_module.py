import warnings
import re
import string
import nltk
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from sentence_transformers import SentenceTransformer, models
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import models
from mofappanalyser.read_write import filetyper

# Download required NLTK data
nltk.download('punkt')
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class ChemTextPreprocessor:
    """
    Chemistry-aware text preprocessing using ChemDataExtractor, SciSpaCy, and SciBERT tokenizer.

    Supports both classic NLP cleaning with transformer-based tokenization,
    and minimal pre-tokenized inputs for SciBERT embedding.

    Attributes:
        lemmatize (bool): Whether to apply lemmatization.
        remove_stopwords (bool): Whether to remove stopwords.
        min_token_length (int): Minimum length of valid tokens.
    """

    def __init__(self, lemmatize=True, remove_stopwords=True, min_token_length=3,
                 transformer_model="allenai/scibert_scivocab_uncased"):
        self.lemmatize = lemmatize
        self.remove_stopwords = remove_stopwords
        self.min_token_length = min_token_length

        self.nlp = spacy.load("en_core_sci_sm")
        self.stopwords = set(stopwords.words("english")) | self.nlp.Defaults.stop_words

        self.transformer_tokenizer = AutoTokenizer.from_pretrained(transformer_model)
        self.transformer_model = models.Transformer(
            transformer_model,
            model_args={"use_safetensors": True}
        )

    def preprocess_for_transformer(self, text: str):
        """
        Minimal preprocessing for transformer-based embedding (SciBERT).
        Returns tokenized tensor input.
        """
        return self.transformer_tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    def preprocess_for_text(self, text: str):
        """
        Tokenize using transformer tokenizer, then apply optional lemmatization and stopword removal.
        """
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", text)
        text = re.sub(r"\bnan\b", "", text)
        text = re.sub(r"[“”]", '', text)
        text = text.translate(str.maketrans('', '', string.punctuation.replace('+', '')))

        tokens = self.transformer_tokenizer.tokenize(text)

        if self.lemmatize:
            doc = self.nlp(" ".join(tokens))
            tokens = [
                token.lemma_ for token in doc
                if token.is_alpha and
                (not self.remove_stopwords or token.lemma_ not in self.stopwords) and
                len(token.lemma_) >= self.min_token_length
            ]
        else:
            tokens = [
                token for token in tokens
                if token.isalpha() and
                (not self.remove_stopwords or token not in self.stopwords) and
                len(token) >= self.min_token_length
            ]

        return tokens

class BERTTopicModeler:
    """
    A wrapper for BERTopic using SciBERT (loaded safely), custom vectorizer, and coherence evaluation.
    """

    def __init__(self):
        model_name = "allenai/scibert_scivocab_uncased"

        word_embedding_model = models.Transformer(
            model_name,
            model_args={"use_safetensors": True}
        )
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        embedding_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 3), min_df=5)
        self.topic_model = BERTopic(embedding_model=embedding_model, vectorizer_model=vectorizer_model)
        self.documents = None

    def preprocess(self, text: str) -> str:
        """
        Tokenizes, lemmatizes, and filters the input text for topic modeling.

        Args:
            text (str): Input text.

        Returns:
            str: Preprocessed and space-joined token string.
        """
        text = str(text).lower()
        text = re.sub(r"\bnan\b", "", text)
        text = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", text)
        text = re.sub(r"[“”]", '', text)
        text = re.sub(r"\s+", " ", text)
        return text

    def fit_transform(self, df: pd.DataFrame, text_column: str = 'text'):
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame.")

        df[text_column] = df[text_column].apply(self.preprocess)

        self.documents = df[text_column].astype(str).tolist()

        topics, probs = self.topic_model.fit_transform(self.documents)
        df_result = df.copy()
        df_result['topic'] = topics
        df_result['topic_prob'] = probs
        return df_result, self.topic_model

    def compute_coherence_values(self, start=5, limit=50, step=5, coherence='c_v'):
        """
        Compute coherence values for different numbers of topics using BERTopic reduction.

        Returns:
            nr_topics_list: List of topic numbers
            coherence_values: Corresponding coherence scores
        """
        if self.documents is None:
            raise ValueError("You must call `fit_transform()` first to populate documents.")

        tokenized_docs = [doc.lower().split() for doc in self.documents]
        dictionary = Dictionary(tokenized_docs)
        corpus = [dictionary.doc2bow(text) for text in tokenized_docs]

        coherence_values = []
        nr_topics_list = []

        for nr_topics in range(start, limit + 1, step):
            reduced_model = self.topic_model.reduce_topics(self.documents, nr_topics=nr_topics)
            topics = reduced_model.get_topics()
            topic_words = [[word for word, _ in topics[i]] for i in topics if topics[i]]

            cm = CoherenceModel(
                topics=topic_words,
                texts=tokenized_docs,
                dictionary=dictionary,
                coherence=coherence
            )
            coherence_score = cm.get_coherence()
            coherence_values.append(coherence_score)
            nr_topics_list.append(nr_topics)

        return nr_topics_list, coherence_values

    def plot_coherence(self, nr_topics_list, coherence_values):
        """Helper method to plot coherence vs nr_topics."""
        plt.figure(figsize=(8, 5))
        plt.plot(nr_topics_list, coherence_values, marker='o')
        plt.xlabel("Number of Topics")
        plt.ylabel("Coherence Score")
        plt.title("Topic Coherence Optimization (BERTopic)")
        plt.grid(True)
        plt.show()

data = filetyper.load_data('../../data/mofs_articles_cleaned.csv')
df = data.sample(n=5000, random_state=42)

modeler = BERTTopicModeler()
df_topics, topic_model = modeler.fit_transform(df, text_column="text")
nr_topics_list, coherence_values = modeler.compute_coherence_values(start=5, limit=50, step=5, coherence='c_v')
modeler.plot_coherence(nr_topics_list, coherence_values)
print(df_topics[['text', 'topic']].head())
topic_model.visualize_topics().show()

