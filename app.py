"""
    Master 2 Health Data Science 2022-2024
    Sujet de NLP - Naima Oubenali
    
    Groupe 8 : Traduction automatique d’abstracts PubMed et extraction des mots clés
    Utiliser des modèles de traduction automatique pour traduire des abstracts PubMed dans différentes langues (3 minimum) 
    et l’extraction/ génération des mots clés qui décriraient le mieux le contenu des abstracts.

    
    Groupe composé de : 
    - Tracy LURANT
    - Antoine TESTON
    - Malik MAMMAR
    
    TO DO : 
    REQUIREMENT.TXT
    README.MD
    DOCSTRING des fonctions

    Fonctionnalités :
    Webscraping
    NLP - BERT
    Extraction des mots clés
    
    Concentration sur les domaines Données synthetic, Digital Twins, Mirors cohorts
"""

# Importation des libraries
import streamlit as st
import requests
from bs4 import BeautifulSoup
from keybert import KeyBERT
from transformers import MarianMTModel, MarianTokenizer, pipeline
from nltk.translate.bleu_score import sentence_bleu
import string
import deepl
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import numpy as np


#-----------------------------------------------



DEEPL_API_KEY = "8c7e317d-7ac9-46ed-9f1f-dc7b0a90cd30:fx"

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def search_pubmed(query, max_results=10):
    """
    Recherche des articles sur PubMed en fonction d'une requête donnée.

    Args:
        query (str): La requête de recherche à utiliser pour trouver des articles sur PubMed.
        max_results (int): Le nombre maximum de résultats à retourner (par défaut 10).

    Returns:
        list: Une liste d'identifiants d'articles PubMed correspondant à la requête.
    """
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={query}&retmax={max_results}&retmode=json"
    response = requests.get(url)
    results = response.json()
    id_list = results.get('esearchresult', {}).get('idlist', [])
    return id_list

def get_pubmed_article_details(pubmed_id):
    """
    Obtient les détails d'un article PubMed en utilisant son identifiant.

    Args:
        pubmed_id (str): L'identifiant de l'article PubMed.

    Returns:
        tuple: Un tuple contenant le titre, le résumé et les mots clés de l'article.
    """
    url = f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    title = soup.find('h1', {'class': 'heading-title'})
    if title:
        title = title.get_text(strip=True)
    else:
        title = "Titre non trouvé"

    abstract_div = soup.find('div', {'class': 'abstract-content'})
    if abstract_div:
        abstract = abstract_div.get_text(strip=True)
    else:
        abstract = "Abstract non trouvé"

    return title, abstract

def extract_keywords_keybert(text, num_keywords=5):
    """
    Extrait les mots clés d'un texte donné en utilisant KeyBERT.

    Args:
        text (str): Le texte à analyser pour extraire des mots clés.
        num_keywords (int): Le nombre de mots clés à extraire (par défaut 5).

    Returns:
        list: Une liste des mots clés extraits du texte.
    """
    kw_model = KeyBERT(model='all-MiniLM-L6-v2')
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words=None, top_n=num_keywords)
    return [kw[0] for kw in keywords]

def preprocess_abstract_tfidf(text):
    """
    Prétraite un texte donné pour l'analyse TF-IDF.

    Args:
        text (str): Le texte à prétraiter.

    Returns:
        str: Le texte prétraité.
    """
    # Suppression des espaces inutiles
    review_text = text.strip()
    # Conversion en minuscules
    review_text = review_text.lower()
    # Suppression de la ponctuation
    review_text = review_text.translate(str.maketrans('', '', string.punctuation))
    # Tokenisation
    tokens = word_tokenize(review_text)
    # Suppression des stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Stemming
    porter = PorterStemmer()
    stemmed_tokens = [porter.stem(word) for word in tokens]
    # Lemmatisation
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in stemmed_tokens]
    # Retourne le texte prétraité
    return ' '.join(lemmatized_words)

def extract_keywords_tfidf(text, num_keywords=5):
    """
    Extrait les mots clés d'un texte donné en utilisant TF-IDF.

    Args:
        text (str): Le texte à analyser pour extraire des mots clés.
        num_keywords (int): Le nombre de mots clés à extraire (par défaut 5).

    Returns:
        list: Une liste des mots clés extraits du texte.
    """
    cleaned_text = preprocess_abstract_tfidf(text)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([cleaned_text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    sorted_indices = tfidf_scores.argsort()[::-1]
    top_keywords = feature_names[sorted_indices][:num_keywords]
    return top_keywords.tolist()

def split_text(text):
    """
    Sépare le texte en phrases, chaque phrase formant un chunk.

    Args:
        text (str): Le texte à séparer.

    Returns:
        list: Une liste de chunks de texte, où chaque chunk est une phrase complète.
    """
    # Tokenisation en phrases
    sentences = nltk.sent_tokenize(text)
    
    return sentences

def translate_with_deepl(text, tgt_lang):
    translator = deepl.Translator(DEEPL_API_KEY)
    result = translator.translate_text(text, target_lang=tgt_lang)
    return result['translations'][0]['text']

def translate_text(text, src_lang, tgt_lang, model_choice):
    if model_choice == 'MarianMT':
        model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        chunks = split_text(text)
        translated_chunks = []
        for chunk in chunks:
            translated = model.generate(**tokenizer(chunk, return_tensors="pt", padding=True))
            tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
            translated_chunks.append(tgt_text[0])
        return ' '.join(translated_chunks)
    
    elif model_choice == 'GoogleT5':
        translation_pipeline = pipeline(f'translation_{src_lang}_to_{tgt_lang}', model='t5-base', tokenizer='t5-base')
        chunks = split_text(text)
        translated_chunks = []
        for chunk in chunks:
            translated = translation_pipeline(chunk, max_length=500, truncation=True)
            translated_chunks.append(translated[0]['translation_text'])
        return ' '.join(translated_chunks)
    
    else:
        return "Modèle de traduction non supporté"

def calculate_bleu(reference_text, tgt_lang, candidate_text):
    reference = [translate_with_deepl(reference_text, tgt_lang).split()]
    candidate = candidate_text.split()
    score = sentence_bleu(reference, candidate)
    return score
















#----------------------------------------


# Dictionnaire pour mapper les noms des langues aux codes de langue
language_dict = {
    "Français": "fr",
    "Roumain": "ro",
    "Allemand": "de"
}

# Initialiser les variables de session si elles n'existent pas
if 'translated_abstract_marianmt' not in st.session_state:
    st.session_state.translated_abstract_marianmt = ""
if 'translated_abstract_googlet5' not in st.session_state:
    st.session_state.translated_abstract_googlet5 = ""
if 'bleu_score_marianmt' not in st.session_state:
    st.session_state.bleu_score_marianmt = None
if 'bleu_score_googlet5' not in st.session_state:
    st.session_state.bleu_score_googlet5 = None


# Interface utilisateur de Streamlit
st.sidebar.title("Recherche PubMed")
query = st.sidebar.text_input("Entrez votre recherche PubMed, PMID ou DOI:", "machine learning cancer")
max_results = st.sidebar.number_input("Nombre maximal de résultats:", min_value=1, max_value=100, value=10)

# Variable pour le retour en arrière
if 'back' not in st.session_state:
    st.session_state.back = False

# Gestion de l'état de retour en arrière
def go_back():
    st.session_state.back = True

if st.sidebar.button("Rechercher"):
    st.session_state.back = False
    with st.spinner('Recherche en cours...'):
        pubmed_ids = search_pubmed(query, max_results)
        articles = [get_pubmed_article_details(pubmed_id) for pubmed_id in pubmed_ids]

        if articles:
            st.session_state.articles = articles
            st.session_state.pubmed_ids = pubmed_ids
        else:
            st.session_state.articles = []
            st.session_state.pubmed_ids = []

# Affichage des résultats
if "articles" in st.session_state and not st.session_state.back:
    articles = st.session_state.articles
    pubmed_ids = st.session_state.pubmed_ids

    if articles:
        st.sidebar.success(f"Trouvé {len(articles)} articles.")
        for idx, (title, abstract) in enumerate(articles):
            if st.sidebar.button(title, key=pubmed_ids[idx]):
                st.session_state["selected_pubmed_id"] = pubmed_ids[idx]
    else:
        st.sidebar.error("Aucun article trouvé.")

# Affichage de l'article sélectionné
if "selected_pubmed_id" in st.session_state and not st.session_state.back:
    pubmed_id = st.session_state["selected_pubmed_id"]
    title, abstract = get_pubmed_article_details(pubmed_id)
    
    st.markdown("---")
    st.header(title)
    st.subheader("Abstract")
    st.write(abstract)

    st.subheader("Mots Clés Générés")
    keybert_keywords = extract_keywords_keybert(abstract, 5)
    tfidf_keywords = extract_keywords_tfidf(abstract, 5)
    
    tab1, tab2 = st.tabs(["KeyBERT", "TF-IDF"])
    
    with tab1:
        st.write("Mots clés extraits avec KeyBERT :")
        st.write(keybert_keywords)
    
    with tab2:
        st.write("Mots clés extraits avec TF-IDF :")
        st.write(tfidf_keywords)
        
    st.subheader("Traduction")
    language_dict = {"Français": "fr", "Espagnol": "es", "Allemand": "de", "Italien": "it"}
    tgt_lang_display = st.selectbox("Choisir la langue de traduction", list(language_dict.keys()))
    tgt_lang = language_dict[tgt_lang_display]
    
    if st.button("Traduire"):
        with st.spinner("Traduction en cours..."):
            st.session_state.translated_abstract_marianmt = translate_text(abstract, 'en', tgt_lang, "MarianMT")
            st.session_state.translated_abstract_googlet5 = translate_text(abstract, 'en', tgt_lang, "GoogleT5")
        
    if st.session_state.translated_abstract_marianmt and st.session_state.translated_abstract_googlet5:
        tab1, tab2 = st.tabs(["MarianMT", "GoogleT5"])
        
        with tab1:
            st.markdown("### Traduction avec MarianMT")
            st.write(st.session_state.translated_abstract_marianmt)
            if st.button("Scoring de Traductions (MarianMT)", key="MarianMT"):
                bleu_score_marianmt = calculate_bleu(abstract, tgt_lang, st.session_state.translated_abstract_marianmt)
                st.session_state.bleu_score_marianmt = bleu_score_marianmt
        
        with tab2:
            st.markdown("### Traduction avec GoogleT5")
            st.write(st.session_state.translated_abstract_googlet5)
            if st.button("Scoring de Traductions (GoogleT5)", key="GoogleT5"):
                bleu_score_googlet5 = calculate_bleu(abstract, tgt_lang, st.session_state.translated_abstract_googlet5)
                st.session_state.bleu_score_googlet5 = bleu_score_googlet5

        if st.session_state.bleu_score_marianmt is not None:
            with tab1:
                st.write(f"Score BLEU: {st.session_state.bleu_score_marianmt}")
        
        if st.session_state.bleu_score_googlet5 is not None:
            with tab2:
                st.write(f"Score BLEU: {st.session_state.bleu_score_googlet5}")
