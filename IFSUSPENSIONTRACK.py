import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import io
import requests
import os
# import traceback

# --- Configuration de la Page Streamlit ---
st.set_page_config(
    page_title="Analyseur Sécurité Alimentaire IFS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Personnalisé pour un Design "Slick" (Thème Sombre) ---
custom_css = """
<style>
    /* --- Global --- */
    body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
        color: #E0E2E8; /* Texte principal plus clair */
    }
    .stApp {
        background-color: #0E1117; /* Fond principal très sombre */
    }
    h1, h2, h3, h4, h5, h6 {
        color: #A2FACF; /* Vert menthe pour les titres */
    }

    /* --- Sidebar --- */
    .css-1d391kg, .css-1lcbmhc { /* Classes de la Sidebar */
        background-color: #1A1F2B; /* Fond sidebar un peu plus clair */
        border-right: 1px solid #30363F;
    }
    .css-1d391kg .stRadio > label, .css-1d391kg .stFileUploader > label, .css-1d391kg .stButton > button {
        color: #C9D1D9; /* Texte des labels et boutons sidebar */
    }
    .css-1d391kg .stButton > button {
        background-color: #238636; /* Vert GitHub pour les boutons */
        border: none;
        border-radius: 6px;
        transition: background-color 0.2s ease-in-out;
    }
    .css-1d391kg .stButton > button:hover {
        background-color: #2EA043;
    }
    .css-1d391kg .stImage > img { /* Logo IFS */
        background-color: white; /* Fond blanc pour le logo SVG */
        padding: 5px;
        border-radius: 6px;
    }


    /* --- Onglets --- */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #161B22; /* Fond de la barre d'onglets */
        border-radius: 8px 8px 0 0;
        border-bottom: 1px solid #30363F;
        padding: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        color: #8B949E; /* Couleur des onglets inactifs */
        border-radius: 6px;
        margin: 0 0.25rem;
        padding: 0.5rem 1rem;
        transition: background-color 0.2s ease-in-out, color 0.2s ease-in-out;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #21262D;
        color: #C9D1D9;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0D1117; /* Fond de l'onglet actif (correspond au fond principal) */
        color: #A2FACF !important; /* Vert menthe pour l'onglet actif */
        border-bottom: 3px solid #A2FACF;
        box-shadow: none;
    }

    /* --- Expanders --- */
    .stExpander {
        border: 1px solid #30363F;
        border-radius: 8px;
        margin-bottom: 1.2rem;
        background-color: #161B22; /* Fond de l'expander */
    }
    .stExpander header {
        background-color: transparent; /* Enlever le fond par défaut du header */
        color: #C9D1D9;
        font-weight: 600;
        padding: 0.75rem 1rem;
        border-bottom: 1px solid #30363F; /* Ligne de séparation */
    }
    .stExpander header:hover {
        background-color: #21262D;
    }
    .stExpander div[data-testid="stExpanderDetails"] { /* Contenu de l'expander */
        background-color: #0D1117; /* Fond du contenu (correspond au fond principal) */
        padding: 1rem;
        border-radius: 0 0 7px 7px;
    }

    /* --- Métriques --- */
    div[data-testid="stMetric"] {
        background-color: #161B22;
        border: 1px solid #30363F;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetric"] > label { /* Label de la métrique */
        color: #8B949E;
    }
    div[data-testid="stMetric"] > div:nth-child(2) > div { /* Valeur de la métrique */
        color: #A2FACF;
        font-size: 2em;
        font-weight: 600;
    }


    /* --- DataFrames --- */
    .stDataFrame {
        border: 1px solid #30363F;
        border-radius: 8px;
    }
    /* Pour un meilleur contraste dans les tables en thème sombre */
    .dataframe > thead > tr > th { background-color: #21262D; color: #C9D1D9; border-bottom: 1px solid #484F58; }
    .dataframe > tbody > tr > td { border-color: #30363F; }
    .dataframe > tbody > tr:nth-child(even) { background-color: #181c23; } /* Lignes paires légèrement différentes */

    /* --- Cartes Plotly --- */
    .plotly-graph-div {
        border-radius: 8px;
        /* background-color: #161B22 !important; Fond pour les graphiques */
    }

    /* Masquer le "Made with Streamlit" et le menu hamburger (optionnel, mais pour un look "pro") */
    /* footer {visibility: hidden;} */
    /* #MainMenu {visibility: hidden;} */

</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# --- Classe IFSAnalyzer (copiez-collez l'intégralité de la classe ici) ---
# Assurez-vous que la classe IFSAnalyzer est la version la plus récente que nous avons développée,
# avec les corrections pour la checklist et la classification thématique améliorée.
# Pour la brièveté, je ne la répète pas ici, mais elle est CRUCIALE.
# Remplacez le commentaire ci-dessous par le code de votre classe IFSAnalyzer.

# >>> DEBUT DE LA CLASSE IFSAnalyzer (COLLEZ VOTRE CLASSE ICI) <<<
class IFSAnalyzer:
    def __init__(self, locked_file_io, checklist_file_io=None):
        self.locked_df = None
        self.checklist_df = None
        # Définition unique des thèmes et de leurs mots-clés, incluant des numéros de chapitres pertinents
        self.themes_definition = {
            'HYGIENE_PERSONNEL': {
                'text': ['hygien', 'personnel', 'clothing', 'hand wash', 'uniform', 'gmp', 'locker', 'changing room', 'work clothing', 'protective clothes', 'personal items', 'jewellery', 'hair cover', 'beard cover'],
                'chapters': ['3.2.1', '3.2.2', '3.2.3', '3.2.4', '3.2.5', '3.2.6', '3.2.7', '3.2.8', '3.2.9', '3.2.10', '3.4.1', '3.4.2', '3.4.3', '3.4.4']
            },
            'HACCP_CCP_OPRP': {
                'text': ['haccp', 'ccp', 'oprp', 'critical control point', 'hazard analysis', 'validation haccp', 'monitoring procedure', 'corrective action ccp'],
                'chapters': ['2.1.1.1', '2.1.2.1', '2.1.3.1', '2.2.3.8', '2.3.1', '2.3.2.1', '2.3.3.1', '2.3.4.1', '2.3.6.1', '2.3.7.1', '2.3.8.1', '2.3.9.1', '2.3.9.2', '2.3.9.3', '2.3.9.4', '2.3.10.1', '2.3.11.1', '2.3.12.1', '2.3.12.2', '5.3.2']
            },
            'TRACEABILITY': {
                'text': ['traceability', 'trace', 'batch record', 'lot identification', 'identification system'],
                'chapters': ['4.18.1', '4.18.2', '4.18.3', '4.18.4', '4.18.5']
            },
            'ALLERGEN_MANAGEMENT': {
                'text': ['allergen', 'allergy', 'cross-contamination allergen', 'gluten', 'lactose', 'celery', 'mustard', 'wheat', 'egg', 'allergen control', 'allergen labelling'],
                'chapters': ['4.19.1', '4.19.2', '4.19.3', '4.19.4']
            },
            'PEST_CONTROL': {
                'text': ['pest', 'rodent', 'insect', 'trap', 'bait', 'infestation', 'fly', 'mouse', 'rat', 'moth', 'weevil', 'spider', 'cobweb', 'pest management', 'pest monitoring'],
                'chapters': ['4.13.1', '4.13.2', '4.13.3', '4.13.4', '4.13.5', '4.13.6', '4.13.7']
            },
            'CLEANING_SANITATION': {
                'text': ['clean', 'sanitation', 'disinfect', 'cleaning chemical', 'cleaning plan', 'dirt', 'residue', 'hygienic condition', 'cleaning validation'],
                'chapters': ['4.10.1', '4.10.2', '4.10.3', '4.10.4', '4.10.5', '4.10.6', '4.10.7', '4.10.8', '4.10.9']
            },
            'TEMPERATURE_CONTROL': {
                'text': ['temperature', 'cold chain', 'heat treatment', 'refrigerat', 'freez', 'thaw', 'cooling process', 'temperature monitoring'],
                'chapters': ['4.9.9.1', '4.9.9.2', '4.9.9.3', '4.11.1', '4.11.2', '4.11.3', '4.11.4', '4.11.5', '5.4.3']
            },
            'MAINTENANCE_EQUIPMENT_INFRASTRUCTURE': {
                'text': ['maintenance', 'equipment condition', 'calibrat', 'repair', 'infrastructure', 'facility', 'structure', 'conveyor belt', 'building fabric', 'wall', 'floor', 'ceiling', 'drain'],
                'chapters': ['3.1.1', '3.3.1', '4.1.1', '4.1.2', '4.1.3', '4.4.1', '4.4.2', '4.4.3', '4.4.4', '4.4.5', '4.4.6', '4.4.7', '4.4.8', '4.9.1.1', '4.9.2.1', '4.9.2.2', '4.9.2.3', '4.9.3.1', '4.9.4.1', '4.9.5.1', '4.9.6.1', '4.9.6.2', '4.9.7.1', '4.9.8.1', '4.16.1', '4.16.2', '4.16.3', '4.16.4', '4.16.5', '4.17.1', '4.17.2', '4.17.3', '4.17.4', '4.17.5']
            },
            'DOCUMENTATION_RECORDS_PROCEDURES': {
                'text': ['document', 'procedure', 'record keeping', 'manual', 'specification', 'not documented', 'missing record', 'incomplete documentation'],
                'chapters': ['1.3.1', '1.3.2', '2.1.2.1', '2.2.1.1', '2.2.2.1', '2.2.3.1', '2.2.3.2', '2.2.3.9', '2.3.5.1', '4.2.2.1', '5.1.1', '5.1.2', '5.3.1']
            },
            'FOREIGN_BODY_CONTAMINATION': {
                'text': ['foreign body', 'foreign material', 'glass control', 'metal detection', 'x-ray', 'physical contamination', 'wood splinter', 'plastic piece', 'paper scrap', 'paint flake', 'ink migration'],
                'chapters': ['4.12.1', '4.12.2', '4.12.3', '4.12.4', '4.12.5', '4.12.6', '4.12.7', '4.12.8', '4.12.9']
            },
            'STORAGE_WAREHOUSING_TRANSPORT': {
                'text': ['storage condition', 'warehouse practice', 'stock rotation', 'segregation', 'pallet condition', 'raw material storage', 'transport condition', 'loading', 'unloading'],
                'chapters': ['4.14.1', '4.14.2', '4.14.3', '4.14.4', '4.14.5', '4.14.6', '4.15.1', '4.15.2', '4.15.3', '4.15.4', '4.15.5', '4.15.6']
            },
            'SUPPLIER_RAW_MATERIAL_CONTROL': {
                'text': ['supplier approval', 'vendor management', 'purchase specification', 'raw material quality', 'ingredient control', 'packaging material conformity', 'declaration of conformity', 'doc'],
                'chapters': ['4.5.1', '4.5.2', '4.5.3', '4.6.1', '4.6.2', '4.6.3', '4.6.4', '4.7.1.1', '4.7.1.2', '4.2.1.2', '4.2.1.3']
            },
            'LABELLING_PRODUCT_INFORMATION': {
                'text': ['label accuracy', 'labelling requirement', 'product declaration', 'ingredient list error', 'mrl issue', 'allergen declaration', 'nutritional information accuracy'],
                'chapters': ['4.2.1.1', '4.3.1', '4.3.2', '4.3.3', '4.3.4', '4.3.5']
            },
            'QUANTITY_CONTROL_WEIGHT_MEASUREMENT': {
                'text': ['quantity control', 'net weight', 'filling accuracy', 'scale calibration', 'metrological verification', 'underfilling', 'measurement system'],
                'chapters': ['5.4.1', '5.4.2', '5.4.3', '5.5.1', '5.5.2', '5.10.3']
            },
            'MANAGEMENT_SYSTEM_RESPONSIBILITY_CULTURE': {
                'text': ['management responsibility', 'food safety policy', 'quality policy', 'food safety culture', 'internal audit', 'corrective action plan', 'preventive action', 'employee training', 'resource management', 'management review', 'complaint handling'],
                'chapters': ['1.1.1', '1.1.2', '1.2.1', '1.2.2', '1.2.3', '1.2.4', '1.3.1', '1.3.2', '1.4.1', '1.4.2', '3.1.1', '3.1.2', '3.3.1', '3.3.2', '3.3.3', '3.3.4', '4.8.1', '4.8.2', '4.8.4', '4.8.5', '4.9.4', '5.1.1', '5.1.2', '5.1.3', '5.2.1', '5.6.1', '5.6.2', '5.7.1', '5.7.2', '5.8.1', '5.9.1', '5.9.2', '5.9.3', '5.10.1', '5.10.2', '5.11.1', '5.11.2', '5.11.3', '5.11.4']
            },
            'ADMINISTRATIVE_OPERATIONAL_ISSUES': {
                'text': ['payment', 'invoice', 'pay', 'closure', 'discontinued', 'bankrupt', 'denies access', 'auditor access denied', 'site access denied', 'ceased operation', 'fire', 'merged', 'cessation of activity', 'discontinuance of business', 'no longer exists', 'production activity has been stopped', 'terminated the contract', 'renounce'],
                'chapters': []
            },
            'INTEGRITY_PROGRAM_SPECIFIC_ISSUES': {
                'text': ['integrity program audit', 'ifs integrity check', 'on-site check ioc', 'unannounced audit issue'],
                'chapters': []
            }
        }
        self.country_name_mapping = { # Pour la carte choropleth
            "Allemagne": "Germany", "Italie": "Italy", "Pays-Bas": "Netherlands",
            "Espagne": "Spain", "Pologne": "Poland", "France": "France",
            "Belgique": "Belgium", "Autriche": "Austria", "Grèce": "Greece",
            "Turquie": "Turkey", "Danemark": "Denmark", "Royaume-Uni": "United Kingdom",
            "États-Unis d'Amérique": "United States", "USA": "United States", "Chili": "Chile",
            "Brésil": "Brazil", "Maroc": "Morocco", "Bulgarie": "Bulgaria",
            "Roumanie": "Romania", "Lituanie": "Lithuania", "Serbie": "Serbia",
            "Lettonie": "Latvia", "Hongrie": "Hungary", "Irlande": "Ireland",
            "Suisse": "Switzerland", "Portugal": "Portugal", "Norvège": "Norway",
            "République Tchèque": "Czech Republic", "Slovaquie": "Slovakia",
            "Croatie": "Croatia", "Suède": "Sweden", "Finlande": "Finland",
            "Chine": "China", "Inde": "India", "Thaïlande": "Thailand",
             "Égypte": "Egypt", "Afrique du Sud": "South Africa", "Canada": "Canada",
             "Mexique": "Mexico", "Argentine": "Argentina", "Colombie": "Colombia",
             "Pérou": "Peru", "Australie": "Australia", "Nouvelle-Zélande": "New Zealand",
             "Japon": "Japan", "Corée du Sud": "South Korea", "Vietnam": "Vietnam",
             "Malaisie": "Malaysia", "Indonésie": "Indonesia", "Philippines": "Philippines",
             "Singapour": "Singapore", "Émirats Arabes Unis": "United Arab Emirates",
             "Arabie Saoudite": "Saudi Arabia", "Israël": "Israel", "Russie": "Russia",
             "Ukraine": "Ukraine", "Biélorussie": "Belarus", "Kazakhstan": "Kazakhstan",
             "Albanie": "Albania"
            # Ajoutez d'autres pays au besoin
        }
        self.load_data(locked_file_io, checklist_file_io)
        if self.locked_df is not None:
            self.clean_lock_reasons()

    def load_data(self, locked_file_io, checklist_file_io=None):
        try:
            self.locked_df_original = pd.read_csv(locked_file_io, encoding='utf-8')
            if 'Standard' in self.locked_df_original.columns:
                self.locked_df = self.locked_df_original[
                    self.locked_df_original['Standard'].astype(str).str.contains('IFS Food', case=True, na=False)
                ].copy()
                if self.locked_df.empty: # Vérifier si vide APRES filtrage
                    st.warning("Aucune entrée 'IFS Food' trouvée après filtrage. Vérifiez la colonne 'Standard'.")
                    self.locked_df = None # Invalider
                    return
            else:
                st.warning("Colonne 'Standard' non trouvée. Analyse sur toutes les données.")
                self.locked_df = self.locked_df_original.copy()

            if self.locked_df is None: return # S'arrêter si self.locked_df est devenu None

            if checklist_file_io:
                try:
                    temp_checklist_df = pd.read_csv(checklist_file_io, encoding='utf-8')
                    actual_num_col_name = 'NUM_REQ'
                    actual_text_col_name = 'IFS Requirements'
                    target_num_col_name = 'Requirement Number'
                    target_text_col_name = 'Requirement text (English)'

                    # Mappage flexible des colonnes de la checklist
                    if actual_num_col_name in temp_checklist_df.columns and actual_text_col_name in temp_checklist_df.columns:
                        self.checklist_df = temp_checklist_df.rename(columns={
                            actual_num_col_name: target_num_col_name,
                            actual_text_col_name: target_text_col_name
                        })
                    elif target_num_col_name in temp_checklist_df.columns and target_text_col_name in temp_checklist_df.columns:
                        self.checklist_df = temp_checklist_df # Déjà les bons noms
                    else: # Essayer d'autres noms possibles pour la robustesse
                        potential_num_cols = ['Requirement Number', 'NUM_REQ', 'Requirement No.', 'Exigence N°']
                        potential_text_cols = ['Requirement text (English)', 'IFS Requirements', 'Texte Exigence (Anglais)', 'Texte Exigence']
                        
                        found_num_col = next((col for col in potential_num_cols if col in temp_checklist_df.columns), None)
                        found_text_col = next((col for col in potential_text_cols if col in temp_checklist_df.columns), None)

                        if found_num_col and found_text_col:
                            self.checklist_df = temp_checklist_df.rename(columns={
                                found_num_col: target_num_col_name,
                                found_text_col: target_text_col_name
                            })
                            st.info(f"Checklist chargée. Colonnes '{found_num_col}' et '{found_text_col}' mappées.")
                        else:
                            st.warning("Colonnes requises pour la checklist non trouvées. L'analyse des exigences sera limitée.")
                            self.checklist_df = None
                except Exception as e_checklist:
                    st.error(f"Erreur lors du chargement du fichier checklist : {e_checklist}")
                    self.checklist_df = None
        except Exception as e:
            st.error(f"❌ Erreur critique lors du chargement du fichier des suspensions : {e}")
            self.locked_df = None

    def clean_lock_reasons(self):
        if self.locked_df is None or 'Lock reason' not in self.locked_df.columns: return
        self.locked_df['lock_reason_clean'] = self.locked_df['Lock reason'].astype(str).fillna('') \
            .str.lower() \
            .str.replace(r'[\n\r\t]', ' ', regex=True) \
            .str.replace(r'[^\w\s\.\-\/\%§]', ' ', regex=True) \
            .str.replace(r'\s+', ' ', regex=True).str.strip()

    def extract_ifs_chapters(self, text):
        if pd.isna(text) or not isinstance(text, str) or text.strip() == '': return []
        # Patterns plus précis pour éviter les faux positifs (ex: dates)
        patterns = [
            r'(?:ko|major|cl\.|req\.|requirement(?: item)?|chapter|section|point|§|cl\s+|clause)\s*(\d\.\d{1,2}(?:\.\d{1,2})?)(?!\s*\d{2,4})', # Ne pas capturer si suivi d'une date
            r'(\d\.\d{1,2}(?:\.\d{1,2})?)\s*(?:ko|major|:|-|\(ko\)|\(major\))(?!\s*\d{2,4})',
            r'(?<!\d\.)(\d)\s*-\s*ko', # Chapitre principal seul avant - KO (ex: 5 - KO)
            r'requirement\s+(\d\.\d\.\d)(?!\s*\d{2,4})',
            r'cl\s+(\d\.\d+(?:\.\d+)?)(?!\s*\d{2,4})',
            r'§\s*(\d\.\d+(?:\.\d+)?)(?!\s*\d{2,4})',
            r'point\s+(\d\.\d+(?:\.\d+)?)(?!\s*\d{2,4})',
            r'item\s+(\d\.\d+(?:\.\d+)?)(?!\s*\d{2,4})' # Pour "item 1.2.3"
        ]
        chapters_found = []
        normalized_text = text.lower().replace('\n', ' ').replace('\r', ' ')
        for pattern in patterns:
            matches = re.findall(pattern, normalized_text)
            for match in matches:
                chapter_num_match = match if isinstance(match, str) else (match[-1] if isinstance(match, tuple) and match[-1] else match[0] if isinstance(match, tuple) and match[0] else None)
                if chapter_num_match:
                    chapter_num = str(chapter_num_match).strip().rstrip('.').strip()
                    if re.fullmatch(r'\d(\.\d+){0,2}', chapter_num): # Accepte X, X.Y, X.Y.Z
                        main_chapter_part = chapter_num.split('.')[0]
                        if main_chapter_part.isdigit() and 1 <= int(main_chapter_part) <= 6:
                             chapters_found.append(chapter_num)
        return sorted(list(set(chapters_found)))

    def analyze_themes(self):
        if self.locked_df is None or 'lock_reason_clean' not in self.locked_df.columns: return {}, {}
        theme_assignments = []
        if 'Lock reason' not in self.locked_df.columns: # Vérification supplémentaire
             st.error("La colonne 'Lock reason' est manquante dans le fichier des suspensions.")
             return {}, {}

        extracted_chapters_series = self.locked_df['Lock reason'].apply(
            lambda x: self.extract_ifs_chapters(x) if pd.notna(x) else []
        )

        for index, row in self.locked_df.iterrows():
            reason_text_clean = row.get('lock_reason_clean', '')
            original_reason = row.get('Lock reason', '')
            supplier = row.get('Supplier', 'N/A')
            country = row.get('Country/Region', 'N/A')
            extracted_chapters = extracted_chapters_series.loc[index]
            best_theme = 'NON_CLASSIFIE'
            max_score = 0

            admin_theme = 'ADMINISTRATIVE_OPERATIONAL_ISSUES'
            admin_keywords = self.themes_definition.get(admin_theme, {}).get('text', [])
            if any(re.search(r'\b' + re.escape(kw.lower()) + r'\b', reason_text_clean) for kw in admin_keywords):
                best_theme = admin_theme
                max_score = 200 # Priorité haute

            if max_score < 200: # Si ce n'est pas classé comme admin
                for theme_name, theme_data in self.themes_definition.items():
                    if theme_name == admin_theme: continue
                    current_score = 0
                    for chap_kw in theme_data.get('chapters', []):
                        if chap_kw in extracted_chapters: current_score += 100
                    
                    text_match_count = 0
                    for kw in theme_data.get('text', []):
                        pattern = r'\b' + re.escape(kw.lower()) + r'\b' # Mot entier
                        if re.search(pattern, reason_text_clean): text_match_count += 1
                        elif kw.lower() in reason_text_clean: text_match_count += 0.5 # Partiel
                    current_score += text_match_count * 20

                    if current_score > max_score:
                        max_score = current_score
                        best_theme = theme_name
            
            if best_theme != admin_theme and max_score < 20: best_theme = 'NON_CLASSIFIE'

            theme_assignments.append({'theme': best_theme, 'reason': original_reason, 'supplier': supplier, 'country': country})

        final_theme_counts = Counter()
        final_theme_details = {theme_name: [] for theme_name in list(self.themes_definition.keys()) + ['NON_CLASSIFIE']}
        for assignment in theme_assignments:
            final_theme_counts[assignment['theme']] += 1
            final_theme_details[assignment['theme']].append({"reason": assignment['reason'], "supplier": assignment['supplier'], "country": assignment['country']})
        return final_theme_counts, final_theme_details
    
    def geographic_analysis(self):
        if self.locked_df is None or 'Country/Region' not in self.locked_df.columns: return None
        geo_df = self.locked_df.groupby('Country/Region').size().sort_values(ascending=False).reset_index(name='total_suspensions')
        # Appliquer le mappage ici pour la carte choropleth
        geo_df['Country/Region_EN'] = geo_df['Country/Region'].map(self.country_name_mapping).fillna(geo_df['Country/Region'])
        return geo_df

    # ... (clean_product_scopes, product_scope_analysis, chapter_frequency_analysis, analyze_audit_types comme avant)
    # ... (generate_ifs_recommendations_analysis, cross_analysis_scope_themes comme avant)
    # ... (TOUTES les méthodes _create_plotly_... et _create_matplotlib_figure_for_pdf, _add_text_to_pdf_page, export_report_to_pdf)
    # ... (et les generate_..._text DOIVENT ÊTRE COPIÉES ICI INTÉGRALEMENT DE LA VERSION PRÉCÉDENTE)
    # Pour des raisons de concision, je ne les répète pas. Il est crucial de les avoir.
    # Assurez-vous que la classe est complète.

    # Je remets les méthodes qui nécessitaient des ajustements ou qui étaient cruciales
    def clean_product_scopes(self, scope_text): # Reste identique
        if pd.isna(scope_text): return []
        scope_text = str(scope_text)
        raw_scopes = re.split(r'[,;\s"\'’`]|et|\/|&|\.\s', scope_text)
        cleaned_scopes = []
        for scope in raw_scopes:
            scope = scope.strip().replace('"', '').replace("'", "")
            if not scope or not scope.isdigit(): continue
            num = int(scope)
            if 1 <= num <= 11:
                cleaned_scopes.append(str(num))
            elif num > 1000:
                potential_scope_2 = str(num % 100)
                potential_scope_1 = str(num % 10)
                if potential_scope_2 in ['10', '11']: cleaned_scopes.append(potential_scope_2)
                elif potential_scope_1 in [str(i) for i in range(1,10)]: cleaned_scopes.append(potential_scope_1)
        return list(set(cleaned_scopes))

    def product_scope_analysis(self): # Reste identique
        if self.locked_df is None or 'Product scopes' not in self.locked_df.columns: return None
        all_scopes = []
        for scopes_text in self.locked_df['Product scopes'].dropna():
            all_scopes.extend(self.clean_product_scopes(scopes_text))
        return Counter(all_scopes)

    def chapter_frequency_analysis(self): # Reste identique
        if self.locked_df is None or 'Lock reason' not in self.locked_df.columns: return Counter()
        all_chapters = []
        for reason in self.locked_df['Lock reason'].dropna():
            all_chapters.extend(self.extract_ifs_chapters(reason))
        return Counter(all_chapters)

    def analyze_audit_types(self): # Reste identique
        if self.locked_df is None: return {}, {}
        audit_keywords = {
            'INTEGRITY_PROGRAM_IP': ['integrity program', 'integrity', 'programme intégrité', 'programme integrity','onsite check', 'on site check', 'on-site check', 'on-site integrity check', 'ioc', 'i.o.c', 'ip audit', 'integrity audit', 'spot check', 'unannounced audit', 'audit inopiné', 'control inopiné', 'ifs integrity'],
            'SURVEILLANCE_FOLLOW_UP': ['surveillance', 'surveillance audit', 'follow up audit', 'follow-up', 'suivi', 'corrective action'],
            'COMPLAINT_WITHDRAWAL': ['complaint', 'réclamation', 'plainte', 'customer complaint', 'withdrawal', 'retrait', 'recall'],
            'RECERTIFICATION_RENEWAL': ['recertification', 'renewal', 'renouvellement', 're-certification', 'renewal audit']
        }
        audit_analysis = {audit_type: 0 for audit_type in audit_keywords}
        audit_examples = {audit_type: {'examples': [], 'countries': Counter()} for audit_type in audit_keywords}
        for index, row in self.locked_df.iterrows():
            text_to_search = (str(row.get('Lock reason', '')) + " " + str(row.get('Lock history', ''))).lower()
            for audit_type, keywords in audit_keywords.items():
                if any(keyword in text_to_search for keyword in keywords):
                    audit_analysis[audit_type] += 1
                    if len(audit_examples[audit_type]['examples']) < 5:
                        audit_examples[audit_type]['examples'].append({
                            'Supplier': row.get('Supplier', 'N/A'), 'Country/Region': row.get('Country/Region', 'N/A'),
                            'Lock reason': row.get('Lock reason', 'N/A')})
                    audit_examples[audit_type]['countries'][row.get('Country/Region', 'N/A')] += 1
        for audit_type in audit_examples:
            audit_examples[audit_type]['countries'] = dict(audit_examples[audit_type]['countries'].most_common(5))
        return audit_analysis, audit_examples

    def generate_ifs_recommendations_analysis(self):
        if self.locked_df is None or self.checklist_df is None: return None
        if 'Requirement Number' not in self.checklist_df.columns or 'Requirement text (English)' not in self.checklist_df.columns:
            # Message déjà géré dans load_data, mais double-check
            return None
            
        chapter_counts = self.chapter_frequency_analysis()
        if not chapter_counts: return None
        recommendations = []
        for chapter, count in chapter_counts.most_common():
            norm_chapter = chapter.replace("KO ", "").strip()
            req_text_series = self.checklist_df[self.checklist_df['Requirement Number'].astype(str).str.strip() == norm_chapter]['Requirement text (English)']
            req_text = req_text_series.iloc[0] if not req_text_series.empty else "Texte de l'exigence non trouvé dans la checklist."
            recommendations.append({'chapter': chapter, 'count': count, 'requirement_text': req_text})
        return recommendations

    def cross_analysis_scope_themes(self):
        if self.locked_df is None or 'Product scopes' not in self.locked_df.columns or 'lock_reason_clean' not in self.locked_df.columns: return None
        themes_for_cross = {
            'HYGIENE': self.themes_definition['HYGIENE_PERSONNEL']['text'], 'HACCP': self.themes_definition['HACCP_CCP_OPRP']['text'],
            'TRACE': self.themes_definition['TRACEABILITY']['text'], 'ALLERGEN': self.themes_definition['ALLERGEN_MANAGEMENT']['text'],
            'CLEAN': self.themes_definition['CLEANING_SANITATION']['text'], 'MAINT': self.themes_definition['MAINTENANCE_EQUIPMENT_INFRASTRUCTURE']['text'],
            'LABEL': self.themes_definition['LABELLING_PRODUCT_INFORMATION']['text'], 'PEST': self.themes_definition['PEST_CONTROL']['text'],
            'MGT_SYS': self.themes_definition['MANAGEMENT_SYSTEM_RESPONSIBILITY_CULTURE']['text'],
            'F_BODY': self.themes_definition['FOREIGN_BODY_CONTAMINATION']['text']
        }
        scope_theme_data = []
        for idx, row in self.locked_df.iterrows():
            scopes_text, reason_text = row['Product scopes'], row['lock_reason_clean']
            if pd.notna(scopes_text) and pd.notna(reason_text) and reason_text:
                for scope in self.clean_product_scopes(scopes_text):
                    for theme, keywords in themes_for_cross.items():
                        if any(kw.lower() in reason_text for kw in keywords):
                            scope_theme_data.append({'scope': f"Scope {scope}", 'theme': theme})
        if not scope_theme_data: return None
        return pd.DataFrame(scope_theme_data).pivot_table(index='scope', columns='theme', aggfunc='size', fill_value=0)

    def _create_plotly_bar_chart(self, data_dict, title, orientation='v', xaxis_title="", yaxis_title="", color='royalblue', height=400, text_auto=True): # Correction RGBA faite
        if not data_dict : return go.Figure()
        y_data, x_data = (list(data_dict.keys()), list(data_dict.values())) if orientation == 'h' else (list(data_dict.values()), list(data_dict.keys()))
        fig = go.Figure(go.Bar(x=x_data, y=y_data, orientation=orientation, marker_color=color, text=y_data if orientation=='v' else x_data, textposition='auto' if text_auto else None))
        fig.update_layout(title={'text': f"<b>{title}</b>", 'x':0.5, 'font': {'size': 16, 'family': "Arial, sans-serif"}},
                          xaxis_title=xaxis_title, yaxis_title=yaxis_title, height=height,
                          margin=dict(l=20, r=20, t=60, b=20), font=dict(family="Arial, sans-serif", size=10),
                          yaxis=dict(autorange="reversed", tickfont_size=9) if orientation == 'h' else dict(tickfont_size=9),
                          xaxis=dict(tickfont_size=9),
                          plot_bgcolor='rgba(245,245,245,1)', paper_bgcolor='rgba(255,255,255,1)')
        return fig

    def _create_plotly_choropleth_map(self, geo_data_df, title, height=500):
        if geo_data_df is None or geo_data_df.empty or 'Country/Region_EN' not in geo_data_df.columns: return go.Figure()
        fig = px.choropleth(geo_data_df, locations="Country/Region_EN", locationmode='country names',
                            color="total_suspensions", hover_name="Country/Region", # Afficher le nom original au survol
                            color_continuous_scale=px.colors.sequential.Viridis_r,
                            title=title, height=height)
        fig.update_layout(title={'text': f"<b>{title}</b>", 'x':0.5, 'font': {'size': 16, 'family': "Arial, sans-serif"}},
                          geo=dict(showframe=False, showcoastlines=True, projection_type='natural earth', bgcolor='rgba(230,240,255,1)'),
                          margin=dict(l=10, r=10, t=50, b=10), font=dict(family="Arial, sans-serif"),
                          paper_bgcolor='rgba(255,255,255,1)')
        return fig

    def _create_plotly_heatmap(self, pivot_matrix, title, height=500): # Correction RGBA faite
        if pivot_matrix is None or pivot_matrix.empty: return go.Figure()
        fig = px.imshow(pivot_matrix, text_auto='.0f', aspect="auto", color_continuous_scale='Blues', title=title, height=height)
        fig.update_layout(title={'text': f"<b>{title}</b>", 'x':0.5, 'font': {'size': 16, 'family': "Arial, sans-serif"}},
                          margin=dict(l=10, r=10, t=80, b=10), font=dict(family="Arial, sans-serif"),
                          xaxis=dict(tickangle=30, side='bottom', tickfont_size=9), yaxis=dict(tickfont_size=9),
                          paper_bgcolor='rgba(255,255,255,1)')
        fig.update_traces(hovertemplate="Scope: %{y}<br>Thème: %{x}<br>Cas: %{z}<extra></extra>")
        return fig

    def _add_text_to_pdf_page(self, fig, text_lines, start_y=0.95, line_height=0.035, font_size=9, title="", title_font_size=14, max_chars_per_line=100): # Reste identique
        ax = fig.gca(); ax.clear(); ax.axis('off')
        if title:
            ax.text(0.5, start_y, title, ha='center', va='top', fontsize=title_font_size, fontweight='bold', fontname='DejaVu Sans')
            start_y -= (line_height * 2.5)
        current_y = start_y
        for line in text_lines:
            import textwrap
            wrapped_lines = textwrap.wrap(line, width=max_chars_per_line, break_long_words=False, replace_whitespace=False)
            for wrapped_line in wrapped_lines:
                if current_y < 0.05: return False
                fw = 'bold' if line.startswith(tuple(["🎯","📊","🌍","🏭","📋","🔍", "---"])) else 'normal'
                fs = font_size + 1 if fw == 'bold' else font_size
                if line.startswith("---"): fs -=1
                ax.text(0.03, current_y, wrapped_line, ha='left', va='top', fontsize=fs, fontweight=fw, fontname='DejaVu Sans')
                current_y -= line_height
            if not line.strip(): current_y -= (line_height * 0.3)
        return True

    def _create_matplotlib_figure_for_pdf(self, data_dict_or_df, title, x_label="", y_label="", chart_type='barh', top_n=10, color='skyblue', xtick_rotation=0, ytick_fontsize=8): # Reste identique (avec corrections précédentes)
        if not data_dict_or_df and not isinstance(data_dict_or_df, pd.DataFrame) : return None
        fig, ax = plt.subplots(figsize=(10, 6.5))
        items, values = [], []

        if isinstance(data_dict_or_df, (Counter, dict)):
            filtered_data = {k: v for k, v in data_dict_or_df.items() if isinstance(v, (int, float)) and v > 0} # S'assurer que v est numérique
            if not filtered_data: return None
            sorted_data = dict(sorted(filtered_data.items(), key=lambda item: item[1], reverse=True)[:top_n])
            items = [str(k).replace('_',' ').replace('MANAGEMENT','MGMT').replace('RESPONSIBILITY','RESP.')[:35] for k in sorted_data.keys()]
            values = list(sorted_data.values())
        elif isinstance(data_dict_or_df, pd.DataFrame):
            df_top = data_dict_or_df.head(top_n)
            if 'Country/Region' in df_top.columns and 'total_suspensions' in df_top.columns:
                items = df_top['Country/Region'].tolist(); values = df_top['total_suspensions'].tolist(); chart_type = 'bar'
            elif 'chapter' in df_top.columns and 'count' in df_top.columns and 'requirement_text' in df_top.columns:
                 df_top_filtered = df_top[df_top['count'] > 0]
                 if df_top_filtered.empty: return None
                 items = [f"{row['chapter']}\n({str(row['requirement_text'])[:40]}...)" if row['requirement_text'] != "Texte de l'exigence non trouvé dans la checklist fournie." else row['chapter'] for index, row in df_top_filtered.iterrows()]
                 values = df_top_filtered['count'].tolist(); chart_type = 'bar'
            else:
                if not df_top.empty:
                    if len(df_top.columns) >= 1:
                         items = df_top.index.astype(str).tolist() if len(df_top.columns) == 1 else df_top.iloc[:,0].astype(str).tolist()
                         raw_values = df_top.iloc[:,0].tolist() if len(df_top.columns) == 1 else df_top.iloc[:,1].tolist()
                         numeric_values = [v for v in raw_values if isinstance(v, (int, float)) and v > 0]
                         if not numeric_values: return None
                         valid_indices = [i for i, v in enumerate(raw_values) if isinstance(v, (int, float)) and v > 0]
                         items = [items[i] for i in valid_indices][:len(numeric_values)] # Assurer la même longueur
                         values = numeric_values
                    else: return None

        if not items or not values or all(v == 0 for v in values): return None

        if chart_type == 'barh':
            ax.barh(items, values, color=color, edgecolor='grey', zorder=3)
            ax.set_yticklabels(items, fontsize=ytick_fontsize, fontname='DejaVu Sans')
            ax.invert_yaxis()
            ax.set_xlabel(x_label if x_label else 'Nombre de cas', fontsize=10, fontname='DejaVu Sans')
            for i, v_ in enumerate(values): ax.text(v_ + (max(values, default=1)*0.01), i, str(v_), va='center', fontsize=8, fontname='DejaVu Sans', zorder=5)
            ax.set_xlim(0, max(values, default=1) * 1.15)
        elif chart_type == 'bar':
            bars = ax.bar(items, values, color=color, edgecolor='grey', zorder=3)
            ax.set_xticklabels(items, rotation=xtick_rotation, ha='right' if xtick_rotation > 0 else 'center', fontsize=ytick_fontsize, fontname='DejaVu Sans')
            ax.set_ylabel(y_label if y_label else 'Nombre de cas', fontsize=10, fontname='DejaVu Sans')
            for bar_idx, bar_obj in enumerate(bars): # Correction pour itérer sur les barres
                yval = bar_obj.get_height()
                ax.text(bar_obj.get_x() + bar_obj.get_width()/2.0, yval + (max(values, default=1)*0.01), int(yval), ha='center', va='bottom', fontsize=8, fontname='DejaVu Sans', zorder=5)
            ax.set_ylim(0, max(values, default=1) * 1.15)

        ax.set_title(title, fontsize=13, fontweight='bold', pad=20, fontname='DejaVu Sans')
        ax.grid(axis='x' if chart_type == 'barh' else 'y', linestyle=':', alpha=0.6, zorder=0)
        sns.despine(left=True, bottom=True)
        plt.tight_layout(pad=2.0)
        return fig

    def export_report_to_pdf(self, filename='IFS_Analysis_Report.pdf'): # Reste identique (avec corrections précédentes)
        if self.locked_df is None: return None
        try:
            with PdfPages(filename) as pdf:
                total_suspensions = len(self.locked_df)
                if total_suspensions == 0:
                    fig = plt.figure(figsize=(8.5, 11)); self._add_text_to_pdf_page(fig, ["Aucune donnée à analyser."], title="Rapport d'Analyse IFS"); pdf.savefig(fig); plt.close(fig); return filename

                fig = plt.figure(figsize=(8.5, 11))
                ln_o = st.session_state.get('locked_file_name_original', 'N/A')
                cn_o = st.session_state.get('checklist_file_name_original', 'Non fournie')
                title_text = [f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}", "", f"Fichier Suspensions: {ln_o}", f"Fichier Checklist: {cn_o}", "", "📊 VUE D'ENSEMBLE"]
                title_text.append(f"   • Total suspensions IFS Food: {total_suspensions}")
                wr_c = self.locked_df['Lock reason'].notna().sum(); title_text.append(f"   • Avec motifs: {wr_c} ({wr_c/total_suspensions*100:.1f}% si total > 0 else 0%)")
                audit_s_sum, _ = self.analyze_audit_types(); total_as = sum(audit_s_sum.values()); title_text.append(f"   • Liées à audits spécifiques: {total_as} ({total_as/total_suspensions*100:.1f}% si total > 0 else 0%)")
                self._add_text_to_pdf_page(fig, title_text, title="Rapport d'Analyse IFS Food Safety"); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

                tc_full, _ = self.analyze_themes()
                tc_tech = {k:v for k,v in tc_full.items() if k not in ['ADMINISTRATIVE_OPERATIONAL_ISSUES', 'NON_CLASSIFIE']}
                fig_t = self._create_matplotlib_figure_for_pdf(tc_tech, 'Top 10 Thèmes Techniques NC', color='indianred', ytick_fontsize=7);
                if fig_t: pdf.savefig(fig_t, bbox_inches='tight'); plt.close(fig_t)

                gs = self.geographic_analysis(); fig_g = self._create_matplotlib_figure_for_pdf(gs, 'Top 10 Pays', chart_type='bar', color='lightseagreen', xtick_rotation=35, ytick_fontsize=7);
                if fig_g: pdf.savefig(fig_g, bbox_inches='tight'); plt.close(fig_g)
                sc = self.product_scope_analysis(); sc_p = {f"Sc {k}": v for k,v in sc.items()}; fig_s = self._create_matplotlib_figure_for_pdf(sc_p, 'Top 10 Product Scopes', color='cornflowerblue', ytick_fontsize=7);
                if fig_s: pdf.savefig(fig_s, bbox_inches='tight'); plt.close(fig_s)
                reco = self.generate_ifs_recommendations_analysis()
                if reco:
                    df_r = pd.DataFrame(reco); fig_c = self._create_matplotlib_figure_for_pdf(df_r, 'Top 10 Exigences IFS', chart_type='bar', color='gold', xtick_rotation=35, ytick_fontsize=6);
                else:
                    cc_d = self.chapter_frequency_analysis(); fig_c = self._create_matplotlib_figure_for_pdf(cc_d, 'Top 10 Chapitres (Numéros)', chart_type='bar', color='gold', xtick_rotation=35, ytick_fontsize=7);
                if fig_c: pdf.savefig(fig_c, bbox_inches='tight'); plt.close(fig_c)
                cpm = self.cross_analysis_scope_themes()
                if cpm is not None and not cpm.empty:
                    top_n = min(10, len(cpm.index)); scope_tots = cpm.sum(axis=1).sort_values(ascending=False)
                    cpm_f = cpm.loc[scope_tots.head(top_n).index] if len(cpm.index) > top_n else cpm
                    if not cpm_f.empty and cpm_f.shape[0] > 0 and cpm_f.shape[1] > 0:
                        fig_h, ax_h = plt.subplots(figsize=(10, max(5, len(cpm_f.index)*0.7)))
                        sns.heatmap(cpm_f, annot=True, cmap="Blues", fmt='d', ax=ax_h, annot_kws={"size":7}, linewidths=.5, linecolor='grey');
                        ax_h.set_title('Corrélations: Thèmes vs Scopes (Top)', fontsize=13, fontweight='bold', pad=20, fontname='DejaVu Sans')
                        ax_h.tick_params(axis='x', labelsize=8, rotation=35, ha='right'); ax_h.tick_params(axis='y', labelsize=8, rotation=0)
                        plt.tight_layout(pad=2.0); pdf.savefig(fig_h, bbox_inches='tight'); plt.close(fig_h)
                for gen_func, title_str, lh, fs, mcpl in [
                    (self.generate_detailed_theme_analysis_text, "Analyse Thématique Détaillée", 0.03, 8, 110),
                    (self.generate_audit_analysis_report_text, "Analyse des Types d'Audits", 0.03, 8, 110) ]:
                    fig = plt.figure(figsize=(8.5, 11)); text_content = gen_func()
                    self._add_text_to_pdf_page(fig, text_content.splitlines(), title=title_str, line_height=lh, font_size=fs, max_chars_per_line=mcpl)
                    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
                if reco:
                    fig = plt.figure(figsize=(8.5, 11))
                    req_tl = ["Note: Texte de l'exigence de la checklist IFS Food v8 (si fournie).\n"]
                    for r_ in sorted(reco, key=lambda x: x['count'], reverse=True)[:25]:
                        req_tl.extend([f"📋 Chap {r_['chapter']} ({r_['count']} mentions)", f"   Txt: {str(r_['requirement_text'])}", ""]) # Assurer str
                    self._add_text_to_pdf_page(fig, req_tl, title="Détail Exigences IFS", line_height=0.025, font_size=6, max_chars_per_line=130)
                    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
            return filename
        except Exception as e:
            st.error(f"❌ Erreur majeure PDF: {e}")
            # traceback.print_exc()
            return None

    def generate_detailed_theme_analysis_text(self): # Reste identique
        if self.locked_df is None: return ""
        theme_counts, theme_details = self.analyze_themes()
        lines = []
        for theme, count in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                lines.append(f"\n🎯 {theme.replace('_', ' ').title()} ({count} cas)")
                lines.append("-" * 60)
                for i, detail in enumerate(theme_details[theme][:3]):
                    reason_short = str(detail['reason'])[:200] + "..." if len(str(detail['reason'])) > 200 else str(detail['reason'])
                    lines.append(f"   Ex {i+1} ({detail['supplier']}, {detail['country']}):")
                    lines.append(f"     Motif: {reason_short}")
                lines.append("")
        return "\n".join(lines)

    def generate_audit_analysis_report_text(self): # Reste identique
        if self.locked_df is None: return ""
        audit_analysis, audit_examples = self.analyze_audit_types()
        total_suspensions = len(self.locked_df)
        if total_suspensions == 0: return "Aucune suspension à analyser."
        lines = [f"Total audits spécifiques: {sum(audit_analysis.values())} ({sum(audit_analysis.values())/total_suspensions*100:.1f}% du total des suspensions, si total > 0 else 0%)"]
        for audit_type, count in sorted(audit_analysis.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                lines.append(f"\n🎯 {audit_type.replace('_', ' ').title()} ({count} cas - {count/total_suspensions*100:.1f}%)")
                lines.append("-" * 60)
                for i, ex_data in enumerate(audit_examples[audit_type]['examples'][:2]):
                    reason_short = str(ex_data.get('Lock reason', 'N/A'))[:200] + "..." if len(str(ex_data.get('Lock reason', 'N/A'))) > 200 else str(ex_data.get('Lock reason', 'N/A'))
                    lines.append(f"   Ex {i+1} ({ex_data.get('Supplier', 'N/A')}, {ex_data.get('Country/Region', 'N/A')}):")
                    lines.append(f"     Motif: {reason_short}")
                lines.append("")
        return "\n".join(lines)
# >>> FIN DE LA CLASSE IFSAnalyzer <<<


# --- Fonctions Utilitaires pour Streamlit ---
@st.cache_resource
def get_analyzer_instance(_locked_data_io, _checklist_data_io, locked_file_key, checklist_file_key):
    return IFSAnalyzer(_locked_data_io, _checklist_data_io)

@st.cache_data(ttl=3600)
def download_checklist_content_from_github(url):
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        st.error(f"Échec du téléchargement de la checklist depuis GitHub : {e}.")
        return None

# --- Interface Streamlit (main et display_dashboard_tabs) ---
# Copiez ici les fonctions main() et display_dashboard_tabs() de la version précédente du code.
# Assurez-vous qu'elles sont à jour avec les dernières logiques de gestion des fichiers et d'appel à l'analyseur.
# Pour la brièveté, je ne les répète pas ici.

# --- Interface Streamlit ---
def main():
    # (Le code de la fonction main reste identique à la version précédente)
    st.title("🛡️ Analyseur de Sécurité Alimentaire IFS")
    st.markdown("""
    Bienvenue ! Téléversez votre fichier CSV des suspensions IFS (format IFS Locked) pour obtenir une analyse détaillée.
    L'utilisation de la checklist IFS Food V8 (téléchargée depuis GitHub par défaut) enrichira l'analyse des exigences.
    """)

    with st.sidebar:
        st.image("https://www.ifs-certification.com/images/ifs_logo.svg", width=150)
        st.header("⚙️ Options d'Analyse")
        locked_file_uploaded = st.file_uploader("1. Fichier des suspensions IFS (.csv)", type="csv", key="locked_uploader", help="Fichier contenant les données des suspensions (ex: LOCKEDIFS - version OR.csv).")

        st.markdown("---")
        checklist_source = st.radio(
            "2. Source de la Checklist IFS Food V8:",
            ("Utiliser celle de GitHub (Recommandé)", "Téléverser ma checklist", "Ne pas utiliser de checklist"),
            index=0, key="checklist_source_radio",
            help="La checklist permet une analyse plus fine des exigences IFS spécifiques."
        )
        checklist_file_uploaded_ui = None
        if checklist_source == "Téléverser ma checklist":
            checklist_file_uploaded_ui = st.file_uploader("Téléversez votre fichier checklist (.csv)", type="csv", key="checklist_uploader")

    if locked_file_uploaded is not None:
        current_locked_file_key = locked_file_uploaded.name + str(locked_file_uploaded.size)
        current_checklist_file_key = "no_checklist_selected"
        st.session_state.locked_file_name_original = locked_file_uploaded.name

        locked_data_io = io.BytesIO(locked_file_uploaded.getvalue())
        checklist_data_io = None

        if checklist_source == "Téléverser ma checklist" and checklist_file_uploaded_ui is not None:
            checklist_data_io = io.BytesIO(checklist_file_uploaded_ui.getvalue())
            current_checklist_file_key = checklist_file_uploaded_ui.name + str(checklist_file_uploaded_ui.size)
            st.session_state.checklist_file_name_original = checklist_file_uploaded_ui.name
        elif checklist_source == "Utiliser celle de GitHub (Recommandé)":
            checklist_url = "https://raw.githubusercontent.com/M00N69/Action-plan/main/Guide%20Checklist_IFS%20Food%20V%208%20-%20CHECKLIST.csv"
            checklist_text_content = download_checklist_content_from_github(checklist_url)
            if checklist_text_content:
                checklist_data_io = io.StringIO(checklist_text_content)
                current_checklist_file_key = f"github_checklist_content_hash_{hash(checklist_text_content)}"
            else: current_checklist_file_key = "github_checklist_failed"
            st.session_state.checklist_file_name_original = "Checklist IFS Food V8 (GitHub)"
        else: # Ne pas utiliser de checklist
            st.session_state.checklist_file_name_original = "Non fournie"
            current_checklist_file_key = "no_checklist_used"


        analyzer = get_analyzer_instance(locked_data_io, checklist_data_io, current_locked_file_key, current_checklist_file_key)

        if analyzer.locked_df is not None and not analyzer.locked_df.empty:
            st.success(f"Fichier **'{locked_file_uploaded.name}'** analysé : **{len(analyzer.locked_df)}** suspensions IFS Food trouvées.")
            display_dashboard_tabs(analyzer) # La fonction qui affiche les onglets

            st.sidebar.markdown("---")
            st.sidebar.subheader("Téléchargement du Rapport")
            if st.sidebar.button("📄 Générer et Télécharger le PDF", key="pdf_button_main", help="Génère un rapport PDF complet avec graphiques et analyses détaillées."):
                with st.spinner("Génération du rapport PDF en cours... Cela peut prendre quelques instants."):
                    temp_pdf_filename = f"temp_report_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.pdf"
                    pdf_path_generated = analyzer.export_report_to_pdf(filename=temp_pdf_filename)

                    if pdf_path_generated and os.path.exists(pdf_path_generated):
                        with open(pdf_path_generated, "rb") as pdf_file:
                            pdf_bytes = pdf_file.read()
                        st.sidebar.download_button(
                            label="📥 Cliquez ici pour télécharger le PDF",
                            data=pdf_bytes,
                            file_name="Analyse_IFS_Suspensions_Report.pdf",
                            mime="application/pdf",
                            key="download_pdf_main"
                        )
                        st.sidebar.success("Rapport PDF prêt !")
                        try: os.remove(pdf_path_generated)
                        except Exception: pass
                    else:
                        st.sidebar.error("Erreur lors de la création du rapport PDF. Veuillez vérifier les messages d'erreur dans la console ou les logs.")
        else:
            st.error("Aucune donnée IFS Food n'a été trouvée dans le fichier téléversé ou après filtrage. Veuillez vérifier son contenu et son format.")
    else:
        st.info("👈 Veuillez téléverser un fichier CSV des suspensions IFS via la barre latérale pour commencer l'analyse.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("Analyseur IFS v1.3")
    st.sidebar.markdown("Développé par IA")


def display_dashboard_tabs(analyzer):
    # (La fonction display_dashboard_tabs reste la même que dans la version précédente,
    #  elle appelle les méthodes de l'analyseur pour obtenir les données et les afficher.
    #  Assurez-vous de la copier intégralement ici.)
    tab_titles = ["📊 Vue d'Ensemble", "🌍 Géographie", "🏷️ Thèmes Détaillés", "📋 Exigences IFS", "🕵️ Audits Spécifiques", "🔗 Analyse Croisée"]
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_titles)

    with tab1:
        st.header("📊 Vue d'Ensemble des Suspensions")
        if analyzer.locked_df is None or analyzer.locked_df.empty: st.warning("Aucune donnée à afficher."); return
        col1, col2, col3 = st.columns(3)
        total_suspensions = len(analyzer.locked_df)
        with_reasons_count = analyzer.locked_df['Lock reason'].notna().sum()
        audit_analysis_summary, _ = analyzer.analyze_audit_types()
        total_audit_special = sum(audit_analysis_summary.values())

        with col1: st.metric("Total Suspensions IFS Food", total_suspensions, help="Nombre total de suspensions après filtrage pour 'IFS Food'.")
        with col2: st.metric("Avec Motifs Documentés", f"{with_reasons_count} ({with_reasons_count/total_suspensions*100:.1f}% si total > 0 else 0%)", help="Pourcentage de suspensions ayant un motif renseigné.")
        with col3: st.metric("Liées à Audits Spécifiques", f"{total_audit_special} ({total_audit_special/total_suspensions*100:.1f}% si total > 0 else 0%)", help="Suspensions liées à des audits (Integrity Program, surveillance, etc.).")
        st.markdown("---"); st.subheader("Visualisations Clés")
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            theme_counts_full, _ = analyzer.analyze_themes()
            theme_counts_technical = {k: v for k, v in theme_counts_full.items() if k not in ['ADMINISTRATIVE_OPERATIONAL_ISSUES', 'NON_CLASSIFIE']} # Exclure
            if theme_counts_technical:
                top_themes = dict(sorted(theme_counts_technical.items(), key=lambda x:x[1], reverse=True)[:10])
                top_themes_clean = {k.replace('_',' ').replace('MANAGEMENT','MGMT').replace('SYSTEM','SYS').replace('RESPONSIBILITY','RESP.'):v for k,v in top_themes.items() if v > 0}
                if top_themes_clean: st.plotly_chart(analyzer._create_plotly_bar_chart(top_themes_clean, "Top 10 Thèmes Techniques de NC", orientation='h', color='indianred', height=450), use_container_width=True)
        with row1_col2:
            scope_counts = analyzer.product_scope_analysis()
            if scope_counts:
                top_scopes = dict(scope_counts.most_common(10))
                top_scopes_clean = {f"Scope {k}": v for k, v in top_scopes.items() if v > 0}
                if top_scopes_clean: st.plotly_chart(analyzer._create_plotly_bar_chart(top_scopes_clean, "Top 10 Product Scopes Impactés", orientation='h', color='cornflowerblue', height=450), use_container_width=True)

    with tab2:
        st.header("🌍 Analyse Géographique")
        if analyzer.locked_df is None or analyzer.locked_df.empty: st.warning("Aucune donnée géographique."); return
        geo_stats_df = analyzer.geographic_analysis() # Devrait maintenant avoir Country/Region_EN
        if geo_stats_df is not None and not geo_stats_df.empty:
            geo_stats_df_filtered = geo_stats_df[geo_stats_df['total_suspensions'] > 0]
            if not geo_stats_df_filtered.empty:
                st.plotly_chart(analyzer._create_plotly_choropleth_map(geo_stats_df_filtered, "Suspensions par Pays"), use_container_width=True)
                st.markdown("---"); st.subheader("Tableau des Suspensions par Pays (Top 20)")
                # Afficher la colonne originale 'Country/Region' et les suspensions
                display_df_geo = geo_stats_df_filtered[['Country/Region', 'total_suspensions']].head(20)
                st.dataframe(display_df_geo.style.highlight_max(subset=['total_suspensions'], color='rgba(255,170,170,0.5)', axis=0).format({'total_suspensions': '{:,}'}), use_container_width=True)
            else: st.info("Aucun pays avec des suspensions à afficher sur la carte après filtrage.")
        else: st.info("Données géographiques non disponibles.")

    with tab3:
        st.header("🏷️ Analyse Thématique Détaillée")
        if analyzer.locked_df is None or analyzer.locked_df.empty: st.warning("Aucune donnée à afficher."); return
        st.markdown("Explorez les motifs de suspension par thème. Cliquez sur un thème pour voir des exemples.")
        theme_counts, theme_details = analyzer.analyze_themes()
        
        technical_themes = {k:v for k,v in theme_counts.items() if k not in ['ADMINISTRATIVE_OPERATIONAL_ISSUES', 'NON_CLASSIFIE']}
        admin_issues_count = theme_counts.get('ADMINISTRATIVE_OPERATIONAL_ISSUES', 0)
        unclassified_count = theme_counts.get('NON_CLASSIFIE', 0)

        st.subheader("Thèmes Techniques de Non-Conformité")
        for theme, count in sorted(technical_themes.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                with st.expander(f"{theme.replace('_', ' ').title()} ({count} cas)", expanded=False):
                    st.markdown(f"**Exemples de motifs (jusqu'à 5) pour : {theme.replace('_', ' ').title()}**")
                    for i, detail in enumerate(theme_details.get(theme, [])[:5]): # Utiliser .get()
                        st.markdown(f"**Cas {i+1} (Fournisseur: `{detail['supplier']}`, Pays: `{detail['country']}`)**")
                        st.caption(f"{str(detail['reason'])[:600]}...")
                        if i < 4 : st.markdown("---")
        
        if admin_issues_count > 0:
            st.subheader("Problèmes Administratifs / Opérationnels")
            with st.expander(f"Problèmes Administratifs / Opérationnels ({admin_issues_count} cas)", expanded=False):
                for i, detail in enumerate(theme_details.get('ADMINISTRATIVE_OPERATIONAL_ISSUES', [])[:5]):
                    st.markdown(f"**Cas {i+1} (Fournisseur: `{detail['supplier']}`, Pays: `{detail['country']}`)**")
                    st.caption(f"{str(detail['reason'])[:600]}...")
                    if i < 4 : st.markdown("---")
        
        if unclassified_count > 0:
            st.subheader("Motifs Non Classifiés")
            with st.expander(f"Non Classifiés ({unclassified_count} cas)", expanded=False):
                for i, detail in enumerate(theme_details.get('NON_CLASSIFIE', [])[:5]):
                    st.markdown(f"**Cas {i+1} (Fournisseur: `{detail['supplier']}`, Pays: `{detail['country']}`)**")
                    st.caption(f"{str(detail['reason'])[:600]}...")
                    if i < 4 : st.markdown("---")

    with tab4:
        st.header("📋 Analyse des Exigences IFS")
        if analyzer.locked_df is None or analyzer.locked_df.empty: st.warning("Aucune donnée à afficher."); return
        recommendations = analyzer.generate_ifs_recommendations_analysis()
        if recommendations and analyzer.checklist_df is not None:
            st.success("Checklist IFS Food V8 utilisée pour l'analyse des exigences.")
            df_reco = pd.DataFrame(recommendations).sort_values(by='count', ascending=False)
            top_reco_chart_df = df_reco.head(15).copy()
            # S'assurer que 'requirement_text' existe avant de l'utiliser pour le label
            if 'requirement_text' in top_reco_chart_df.columns:
                 top_reco_chart_df['display_label'] = top_reco_chart_df.apply(lambda row: f"{row['chapter']} ({str(row['requirement_text'])[:30]}...)", axis=1)
            else: # Fallback si la colonne manque pour une raison quelconque
                 top_reco_chart_df['display_label'] = top_reco_chart_df['chapter']
            reco_chart_data = pd.Series(top_reco_chart_df['count'].values, index=top_reco_chart_df['display_label']).to_dict()
            st.plotly_chart(analyzer._create_plotly_bar_chart(reco_chart_data, "Top 15 Exigences IFS Citées", orientation='v', color='gold', height=550, text_auto=False), use_container_width=True)
            st.markdown("---"); st.subheader("Détail des Exigences Citées (Top 25)")
            for index, row in df_reco.head(25).iterrows():
                with st.expander(f"Exigence {row['chapter']} ({row['count']} mentions)", expanded=False):
                    st.markdown(f"**Texte de l'exigence :**\n\n> _{str(row['requirement_text'])}_")
        elif recommendations:
             st.warning("Checklist non chargée/valide. Affichage des numéros de chapitres uniquement.")
             df_reco_no_text = pd.DataFrame(recommendations).sort_values(by='count', ascending=False).head(15)
             chapter_counts_dict = pd.Series(df_reco_no_text['count'].values, index=df_reco_no_text['chapter']).to_dict()
             st.plotly_chart(analyzer._create_plotly_bar_chart(chapter_counts_dict, "Top Chapitres IFS Cités (Numéros)", orientation='v', color='gold', height=500), use_container_width=True)
             st.dataframe(df_reco_no_text, use_container_width=True)
        else: st.info("Aucune exigence/chapitre IFS n'a pu être extrait, ou la checklist n'est pas disponible/utilisée.")

    with tab5: # Audits Spécifiques
        # (Reste comme avant)
        st.header("🕵️ Analyse par Audits Spécifiques")
        if analyzer.locked_df is None or analyzer.locked_df.empty: st.warning("Aucune donnée à afficher."); return
        audit_analysis, audit_examples = analyzer.analyze_audit_types()
        if audit_analysis:
            audit_analysis_clean = {k.replace('_', ' ').title():v for k,v in audit_analysis.items() if v > 0}
            if audit_analysis_clean: st.plotly_chart(analyzer._create_plotly_bar_chart(audit_analysis_clean, "Répartition par Type d'Audit Spécifique", color='darkorange', height=400), use_container_width=True)
            st.markdown("---"); st.subheader("Détails et Exemples par Type d'Audit")
            for audit_type, count in sorted(audit_analysis.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    with st.expander(f"{audit_type.replace('_', ' ').title()} ({count} cas)", expanded=False):
                        st.markdown(f"**Exemples (jusqu'à 5) pour : {audit_type.replace('_', ' ').title()}**")
                        for i, ex_data in enumerate(audit_examples[audit_type]['examples'][:5]):
                            st.markdown(f"**Cas {i+1} (Fournisseur: `{ex_data.get('Supplier', 'N/A')}`, Pays: `{ex_data.get('Country/Region', 'N/A')}`)**")
                            st.caption(f"{str(ex_data.get('Lock reason', 'N/A'))[:600]}...")
                            if i < 4 : st.markdown("---")
                        countries_data = audit_examples[audit_type]['countries']
                        if countries_data:
                            st.markdown(f"**Répartition géographique (Top 5 pays) :** {', '.join([f'{c} ({n})' for c, n in countries_data.items()])}")
        else: st.info("Aucune donnée sur les types d'audits spécifiques disponible.")

    with tab6: # Analyse Croisée
        # (Reste comme avant)
        st.header("🔗 Analyse Croisée : Thèmes vs Product Scopes")
        if analyzer.locked_df is None or analyzer.locked_df.empty: st.warning("Aucune donnée à afficher."); return
        cross_pivot_matrix = analyzer.cross_analysis_scope_themes()
        if cross_pivot_matrix is not None and not cross_pivot_matrix.empty:
            top_n_scopes_heatmap = min(15, len(cross_pivot_matrix.index))
            if len(cross_pivot_matrix.index) > top_n_scopes_heatmap:
                scope_totals = cross_pivot_matrix.sum(axis=1).sort_values(ascending=False)
                cross_pivot_matrix_filtered = cross_pivot_matrix.loc[scope_totals.head(top_n_scopes_heatmap).index]
            else: cross_pivot_matrix_filtered = cross_pivot_matrix

            if not cross_pivot_matrix_filtered.empty:
                 st.plotly_chart(analyzer._create_plotly_heatmap(cross_pivot_matrix_filtered, "Fréquence des Thèmes par Product Scope (Top Scopes)", height=max(500, len(cross_pivot_matrix_filtered.index) * 35 + 200)), use_container_width=True)
                 st.markdown("---"); st.subheader("Tableau de Corrélation Complet (Scopes vs Thèmes)")
                 st.dataframe(cross_pivot_matrix.style.background_gradient(cmap='Blues', axis=None).format("{:.0f}"), use_container_width=True)
            else: st.info("Pas assez de données pour la heatmap après filtrage.")
        else: st.info("Données insuffisantes pour l'analyse croisée Thèmes vs Product Scopes.")

# --- Exécution de l'application ---
if __name__ == "__main__":
    main()
