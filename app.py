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

    keywords_section = soup.find('div', {'class': 'keywords-section'})
    if keywords_section:
        keywords = keywords_section.get_text(strip=True).replace('Keywords', '').strip()
    else:
        keywords = "Mots clés non trouvés"

    return title, abstract, keywords

def extract_keywords(text, num_keywords=5):
    """
    Extrait les mots clés d'un texte donné en utilisant KeyBERT.

    Args:
        text (str): Le texte à analyser pour extraire des mots clés.
        num_keywords (int): Le nombre de mots clés à extraire (par défaut 5).

    Returns:
        list: Une liste des mots clés extraits du texte.
    """
    kw_model = KeyBERT(model='all-MiniLM-L6-v2')
    
    
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=num_keywords)
    return [kw[0] for kw in keywords]

def translate_text(text, src_lang, tgt_lang, model_choice):
    """
    Traduit un texte donné d'une langue source vers une langue cible en utilisant un modèle de traduction spécifié.

    Args:
        text (str): Le texte à traduire.
        src_lang (str): Le code de la langue source (ex : 'fr' pour français).
        tgt_lang (str): Le code de la langue cible (ex : 'en' pour anglais).
        model_choice (str): Le choix du modèle de traduction ('MarianMT' ou 'GoogleT5').

    Returns:
        str: Le texte traduit dans la langue cible.
    """
    if model_choice == 'MarianMT':
        model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
        tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        return tgt_text[0]
    
    elif model_choice == 'GoogleT5':
            translation_pipeline = pipeline(f'translation_{src_lang}_to_{tgt_lang}')
            translated = translation_pipeline(text, src_lang=src_lang, tgt_lang=tgt_lang)
            return translated[0]['translation_text']
        
    else:
        return "Modèle de traduction non supporté"

# Interface utilisateur de Streamlit
st.sidebar.title("Recherche PubMed")
query = st.sidebar.text_input("Entrez votre recherche PubMed, PMID ou DOI:", "machine learning cancer")
max_results = st.sidebar.number_input("Nombre maximal de résultats:", min_value=1, max_value=100, value=10)
model_choice = st.sidebar.selectbox("Choix du modèle de traduction", ["MarianMT", "GoogleT5"])
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
        st.success(f"Trouvé {len(articles)} articles.")
        for idx, (title, abstract, keywords) in enumerate(articles):
            if st.button(title, key=pubmed_ids[idx]):
                st.session_state["selected_pubmed_id"] = pubmed_ids[idx]
    else:
        st.error("Aucun article trouvé.")

# Affichage de l'article sélectionné
if "selected_pubmed_id" in st.session_state and not st.session_state.back:
    pubmed_id = st.session_state["selected_pubmed_id"]
    title, abstract, keywords = get_pubmed_article_details(pubmed_id)
    
    st.markdown("---")
    st.header(title)
    st.subheader("Abstract")
    st.write(abstract)

    st.subheader("Mots Clés")
    st.write(keywords)

    st.subheader("Mots Clés Générés")
    extracted_keywords = extract_keywords(abstract)
    st.write(", ".join(extracted_keywords))

    st.subheader("Traduction")
    tgt_lang = st.selectbox("Choisir la langue de traduction", ["fr", "ro", "de"])
    
    if st.button("Traduire"):
        translated_abstract = translate_text(abstract, 'en', tgt_lang, model_choice)
        st.write(translated_abstract)

    if st.button("Retour"):
        go_back()

# Pour exécuter cette application Streamlit, utilisez la commande suivante dans votre terminal:
# streamlit run script_name.py
