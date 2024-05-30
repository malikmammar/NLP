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


from extract import search_pubmed, get_pubmed_article_details, extract_keywords, translate_text


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
