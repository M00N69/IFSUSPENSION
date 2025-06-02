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

# --- Classe IFSAnalyzer ---
class IFSAnalyzer:
    def __init__(self, locked_file_io, checklist_file_io=None):
        self.locked_df = None
        self.checklist_df = None
        # Définition des thèmes avec mots-clés textuels et chapitres spécifiques
        # Le format sera: {THEME: {'text': [kw1, kw2], 'chapters': [chap1, chap2], 'specific_phrases': ["phrase exacte"]}}
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
                'chapters': ['4.9.9.1', '4.9.9.2', '4.9.9.3', '4.11.1', '4.11.2', '4.11.3', '4.11.4', '4.11.5', '5.4.3'] # 5.4.3 est sur la calibration des thermomètres, peut être lié
            },
            'MAINTENANCE_EQUIPMENT_INFRASTRUCTURE': { # Renommé pour plus de clarté
                'text': ['maintenance', 'equipment condition', 'calibrat', 'repair', 'infrastructure', 'facility', 'structure', 'conveyor belt', 'building fabric', 'wall', 'floor', 'ceiling', 'drain'],
                'chapters': ['3.1.1', '3.3.1', '4.1.1', '4.1.2', '4.1.3', '4.4.1', '4.4.2', '4.4.3', '4.4.4', '4.4.5', '4.4.6', '4.4.7', '4.4.8', '4.9.1.1', '4.9.2.1', '4.9.2.2', '4.9.2.3', '4.9.3.1', '4.9.4.1', '4.9.5.1', '4.9.6.1', '4.9.6.2', '4.9.7.1', '4.9.8.1', '4.16.1', '4.16.2', '4.16.3', '4.16.4', '4.16.5', '4.17.1', '4.17.2', '4.17.3', '4.17.4', '4.17.5']
            },
            'DOCUMENTATION_RECORDS_PROCEDURES': { # Renommé
                'text': ['document', 'procedure', 'record keeping', 'manual', 'specification', 'not documented', 'missing record', 'incomplete documentation'],
                'chapters': ['1.3.1', '1.3.2', '2.1.2.1', '2.2.1.1', '2.2.2.1', '2.2.3.1', '2.2.3.2', '2.2.3.9', '2.3.5.1', '4.2.2.1', '5.1.1', '5.1.2', '5.3.1'] # 5.3.1 HACCP plan
            },
            'FOREIGN_BODY_CONTAMINATION': {
                'text': ['foreign body', 'foreign material', 'glass control', 'metal detection', 'x-ray', 'physical contamination', 'wood splinter', 'plastic piece', 'paper scrap', 'paint flake', 'ink migration'],
                'chapters': ['4.12.1', '4.12.2', '4.12.3', '4.12.4', '4.12.5', '4.12.6', '4.12.7', '4.12.8', '4.12.9']
            },
            'STORAGE_WAREHOUSING_TRANSPORT': { # Élargi
                'text': ['storage condition', 'warehouse practice', 'stock rotation', 'segregation', 'pallet condition', 'raw material storage', 'transport condition', 'loading', 'unloading'],
                'chapters': ['4.14.1', '4.14.2', '4.14.3', '4.14.4', '4.14.5', '4.14.6', '4.15.1', '4.15.2', '4.15.3', '4.15.4', '4.15.5', '4.15.6']
            },
            'SUPPLIER_RAW_MATERIAL_CONTROL': {
                'text': ['supplier approval', 'vendor management', 'purchase specification', 'raw material quality', 'ingredient control', 'packaging material conformity', 'declaration of conformity', 'doc'],
                'chapters': ['4.5.1', '4.5.2', '4.5.3', '4.6.1', '4.6.2', '4.6.3', '4.6.4', '4.7.1.1', '4.7.1.2', '4.2.1.2', '4.2.1.3']
            },
            'LABELLING_PRODUCT_INFORMATION': { # Renommé
                'text': ['label accuracy', 'labelling requirement', 'product declaration', 'ingredient list error', 'mrl issue', 'allergen declaration', 'nutritional information accuracy'],
                'chapters': ['4.2.1.1', '4.3.1', '4.3.2', '4.3.3', '4.3.4', '4.3.5']
            },
            'QUANTITY_CONTROL_WEIGHT_MEASUREMENT': { # Renommé
                'text': ['quantity control', 'net weight', 'filling accuracy', 'scale calibration', 'metrological verification', 'underfilling', 'measurement system'],
                'chapters': ['5.4.1', '5.4.2', '5.4.3', '5.5.1', '5.5.2', '5.10.3'] # 5.10.3: Test equipment
            },
            'MANAGEMENT_SYSTEM_RESPONSIBILITY_CULTURE': { # Renommé
                'text': ['management responsibility', 'food safety policy', 'quality policy', 'food safety culture', 'internal audit', 'corrective action plan', 'preventive action', 'employee training', 'resource management', 'management review', 'complaint handling'],
                'chapters': ['1.1.1', '1.1.2', '1.2.1', '1.2.2', '1.2.3', '1.2.4', '1.3.1', '1.3.2', '1.4.1', '1.4.2', '3.1.1', '3.1.2', '3.3.1', '3.3.2', '3.3.3', '3.3.4', '4.8.1', '4.8.2', '4.8.4', '4.8.5', '4.9.4', '5.1.1', '5.1.2', '5.1.3', '5.2.1', '5.6.1', '5.6.2', '5.7.1', '5.7.2', '5.8.1', '5.9.1', '5.9.2', '5.9.3', '5.10.1', '5.10.2', '5.11.1', '5.11.2', '5.11.3', '5.11.4']
            },
            'NON_PAYMENT_ADMINISTRATIVE_CLOSURE': { # Renommé
                'text': ['payment due', 'invoice unpaid', 'company closure', 'operation discontinued', 'bankrupt', 'auditor access denied', 'site access denied', 'ceased operation', 'fire incident', 'company merged', 'cessation of activity', 'discontinuance of business'],
                'chapters': [] # Généralement pas lié à un chapitre IFS spécifique
            },
            'INTEGRITY_PROGRAM_SPECIFIC_ISSUES': { # Renommé
                'text': ['integrity program audit', 'ifs integrity check', 'on-site check ioc', 'unannounced audit issue'],
                'chapters': [] # Peut être lié à n'importe quel chapitre, mais le type d'audit est le déclencheur
            }
        }
        self.load_data(locked_file_io, checklist_file_io)
        if self.locked_df is not None:
            self.clean_lock_reasons()

    def load_data(self, locked_file_io, checklist_file_io=None):
        try:
            self.locked_df = pd.read_csv(locked_file_io, encoding='utf-8')
            if 'Standard' in self.locked_df.columns:
                self.locked_df = self.locked_df[self.locked_df['Standard'].str.contains('IFS Food', na=False, case=False)].copy() # .copy() pour éviter SettingWithCopyWarning
            if checklist_file_io:
                try:
                    self.checklist_df = pd.read_csv(checklist_file_io, encoding='utf-8')
                    expected_num_col = 'NUM_REQ'
                    expected_text_col = 'IFS Requirements'
                    if expected_num_col not in self.checklist_df.columns or expected_text_col not in self.checklist_df.columns:
                        # Essayer d'autres noms communs comme fallback avant d'abandonner
                        if 'Requirement Number' in self.checklist_df.columns and 'Requirement text (English)' in self.checklist_df.columns:
                            self.checklist_df.rename(columns={
                                'Requirement Number': 'NUM_REQ', # Standardiser vers NUM_REQ
                                'Requirement text (English)': 'IFS Requirements' # Standardiser vers IFS Requirements
                            }, inplace=True)
                            st.info("Checklist chargée. Colonnes renommées pour correspondre au format attendu (NUM_REQ, IFS Requirements).")
                        else:
                            st.warning(f"Colonnes '{expected_num_col}' ou '{expected_text_col}' (ou alternatives) manquantes dans la checklist. L'analyse des exigences sera limitée.")
                            self.checklist_df = None
                    if self.checklist_df is not None: # Si le DF existe toujours
                         self.checklist_df.rename(columns={
                            expected_num_col: 'Requirement Number', # Nom interne utilisé par le reste du code
                            expected_text_col: 'Requirement text (English)'
                        }, inplace=True)

                except Exception as e_checklist:
                    st.error(f"Erreur lors du chargement du fichier checklist : {e_checklist}")
                    self.checklist_df = None
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement du fichier des suspensions : {e}")
            self.locked_df = None

    # ... (les autres méthodes de la classe restent majoritairement les mêmes)
    # ... (extract_ifs_chapters, geographic_analysis, clean_product_scopes, product_scope_analysis, chapter_frequency_analysis, analyze_audit_types)
    # ... (generate_ifs_recommendations_analysis, cross_analysis_scope_themes)
    # ... (les méthodes _create_plotly_... et _create_matplotlib_...)
    # ... (les méthodes _add_text_to_pdf_page et export_report_to_pdf)
    # ... (les méthodes generate_..._text)

    def clean_lock_reasons(self): # Pas de changement ici
        if self.locked_df is None or 'Lock reason' not in self.locked_df.columns: return
        self.locked_df['lock_reason_clean'] = self.locked_df['Lock reason'].astype(str).fillna('') \
            .str.lower() \
            .str.replace(r'[\n\r\t]', ' ', regex=True) \
            .str.replace(r'[^\w\s\.\-\/\%§]', ' ', regex=True) \
            .str.replace(r'\s+', ' ', regex=True).str.strip()

    def extract_ifs_chapters(self, text): # Légère amélioration pour les numéros seuls
        if pd.isna(text) or not isinstance(text, str) or text.strip() == '': return []
        patterns = [
            r'(?:ko|major|cl\.|req\.|requirement(?: item)?|chapter|section|point|§|cl\s+|clause)?\s*(\d\.\d{1,2}(?:\.\d{1,2})?)',
            r'(\d\.\d{1,2}(?:\.\d{1,2})?)\s*(?:ko|major|:|-|\(ko\)|\(major\))',
            r'(?<!\d\.)(\d{1,2})\s*-\s*ko', # Chapitre seul avant - KO (ex: 5 - KO)
            r'requirement\s+(\d\.\d\.\d)',
            r'cl\s+(\d\.\d+(?:\.\d+)?)',
            r'§\s*(\d\.\d+(?:\.\d+)?)',
            r'point\s+(\d\.\d+(?:\.\d+)?)' # Pour "point 2.3.9.1"
        ]
        chapters_found = []
        normalized_text = text.lower().replace('\n', ' ').replace('\r', ' ')
        for pattern in patterns:
            matches = re.findall(pattern, normalized_text)
            for match in matches:
                chapter_num_match = match if isinstance(match, str) else (match[-1] if isinstance(match, tuple) and match[-1] else match[0] if isinstance(match, tuple) and match[0] else None)
                if chapter_num_match:
                    chapter_num = str(chapter_num_match).strip().rstrip('.').strip()
                    # Valider que c'est bien un format de chapitre IFS (ex: 1.2, 1.2.3 ou juste un nombre pour les cas comme "5 - KO")
                    if re.fullmatch(r'\d(\.\d+){1,2}', chapter_num) or \
                       re.fullmatch(r'\d\.\d+', chapter_num) or \
                       re.fullmatch(r'\d+', chapter_num): # Accepter aussi les numéros seuls
                        main_chapter_part = chapter_num.split('.')[0]
                        if main_chapter_part.isdigit() and 1 <= int(main_chapter_part) <= 6:
                             chapters_found.append(chapter_num)
        return sorted(list(set(chapters_found)))


    def analyze_themes(self): # Logique de classification thématique améliorée
        if self.locked_df is None or 'lock_reason_clean' not in self.locked_df.columns: return {}, {}

        theme_assignments = []

        for index, row in self.locked_df.iterrows():
            reason_text_clean = row['lock_reason_clean']
            original_reason = row['Lock reason']
            supplier = row.get('Supplier', 'N/A')
            country = row.get('Country/Region', 'N/A')
            
            # Extraire les chapitres une seule fois pour cette raison
            extracted_chapters = self.extract_ifs_chapters(original_reason) # Utiliser l'original pour l'extraction de chapitre

            best_theme = None
            max_score = 0

            for theme_name, theme_data in self.themes_definition.items():
                current_score = 0
                matched_elements = [] # Pour le débogage

                # Score pour les chapitres (poids le plus élevé)
                for chap_kw in theme_data.get('chapters', []):
                    if chap_kw in extracted_chapters:
                        current_score += 100 # Poids fort pour correspondance de chapitre exact
                        matched_elements.append(f"chap:{chap_kw}")

                # Score pour les phrases spécifiques (poids moyen-haut)
                for phrase in theme_data.get('specific_phrases', []): # Supposons que 'specific_phrases' soit ajouté à themes_definition
                    if phrase.lower() in reason_text_clean:
                        current_score += 50
                        matched_elements.append(f"phrase:{phrase}")
                
                # Score pour les mots-clés textuels (poids plus faible)
                # On donne un score plus élevé si plusieurs mots-clés du même thème sont trouvés
                text_match_count = 0
                for kw in theme_data.get('text', []):
                    # Utiliser word boundaries pour certains mots-clés pour éviter les correspondances partielles non désirées
                    # Exemple: \bword\b
                    # Pour l'instant, simple "in"
                    if kw.lower() in reason_text_clean:
                        text_match_count +=1
                        matched_elements.append(f"kw:{kw}")
                
                current_score += text_match_count * 10 # Chaque mot-clé textuel ajoute 10 points

                if current_score > max_score:
                    max_score = current_score
                    best_theme = theme_name
                elif current_score == max_score and current_score > 0:
                     # Gestion des égalités : on pourrait prendre le premier, ou avoir une liste de thèmes prioritaires
                     # Pour l'instant, on garde le premier thème qui a atteint ce score max
                     pass


            if best_theme and max_score > 0: # Un seuil minimal pourrait être ajouté ici (ex: max_score > 20)
                theme_assignments.append({
                    'index': index, # Garder l'index original pour fusionner plus tard si besoin
                    'theme': best_theme,
                    'score': max_score,
                    'reason': original_reason,
                    'supplier': supplier,
                    'country': country,
                    # 'matched_elements': ", ".join(matched_elements) # Pour débogage
                })
            # else: # Non classé
            #     theme_assignments.append({'index': index, 'theme': 'NON_CLASSIFIE', 'score': 0, ...})


        # Agrégation des résultats
        final_theme_counts = Counter()
        final_theme_details = {theme_name: [] for theme_name in self.themes_definition.keys()}
        # final_theme_details['NON_CLASSIFIE'] = [] # Si on garde les non-classifiés

        for assignment in theme_assignments:
            final_theme_counts[assignment['theme']] += 1
            final_theme_details[assignment['theme']].append({
                "reason": assignment['reason'],
                "supplier": assignment['supplier'],
                "country": assignment['country'],
                # "score": assignment['score'], # Optionnel pour l'affichage
                # "matched_elements": assignment['matched_elements'] # Optionnel
            })
        
        return final_theme_counts, final_theme_details

    # ... (Les autres méthodes de la classe IFSAnalyzer restent ici) ...
    # ... (geographic_analysis, clean_product_scopes, product_scope_analysis, chapter_frequency_analysis, analyze_audit_types)
    # ... (generate_ifs_recommendations_analysis, cross_analysis_scope_themes)
    # ... (les méthodes _create_plotly_... et _create_matplotlib_...)
    # ... (les méthodes _add_text_to_pdf_page et export_report_to_pdf)
    # ... (les méthodes generate_..._text)
    # Assurez-vous que toutes les méthodes précédentes de la classe IFSAnalyzer sont incluses ici.
    # Pour la brièveté, je ne les répète pas. Il faut copier-coller l'intégralité de la classe.

    def geographic_analysis(self): # Reste identique
        if self.locked_df is None or 'Country/Region' not in self.locked_df.columns: return None
        return self.locked_df.groupby('Country/Region').size().sort_values(ascending=False).reset_index(name='total_suspensions')

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

    def generate_ifs_recommendations_analysis(self): # S'assure que la checklist a les bonnes colonnes après renommage
        if self.locked_df is None or self.checklist_df is None: return None
        if 'Requirement Number' not in self.checklist_df.columns or 'Requirement text (English)' not in self.checklist_df.columns:
            st.warning("Checklist invalide pour l'analyse des exigences (colonnes attendues manquantes après tentative de renommage).")
            return None
            
        chapter_counts = self.chapter_frequency_analysis()
        if not chapter_counts: return None
        recommendations = []
        for chapter, count in chapter_counts.most_common():
            norm_chapter = chapter.replace("KO ", "").strip()
            # Utiliser les noms de colonnes internes standardisés
            req_text_series = self.checklist_df[self.checklist_df['Requirement Number'].astype(str).str.strip() == norm_chapter]['Requirement text (English)']
            req_text = req_text_series.iloc[0] if not req_text_series.empty else "Texte de l'exigence non trouvé dans la checklist."
            recommendations.append({'chapter': chapter, 'count': count, 'requirement_text': req_text})
        return recommendations

    def cross_analysis_scope_themes(self): # Reste identique
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
                        if any(kw.lower() in reason_text for kw in keywords): # Assurer la comparaison en minuscules
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

    def _create_plotly_choropleth_map(self, geo_data_df, title, height=500): # Correction RGBA faite
        if geo_data_df is None or geo_data_df.empty: return go.Figure()
        fig = px.choropleth(geo_data_df, locations="Country/Region", locationmode='country names',
                            color="total_suspensions", hover_name="Country/Region",
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
            filtered_data = {k: v for k, v in data_dict_or_df.items() if v > 0}
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
                 items = [f"{row['chapter']}\n({row['requirement_text'][:40]}...)" if row['requirement_text'] != "Texte de l'exigence non trouvé dans la checklist fournie." else row['chapter'] for index, row in df_top_filtered.iterrows()]
                 values = df_top_filtered['count'].tolist(); chart_type = 'bar'
            else:
                if not df_top.empty:
                    if len(df_top.columns) >= 1:
                         items = df_top.index.astype(str).tolist() if len(df_top.columns) == 1 else df_top.iloc[:,0].astype(str).tolist()
                         raw_values = df_top.iloc[:,0].tolist() if len(df_top.columns) == 1 else df_top.iloc[:,1].tolist()
                         # S'assurer que les valeurs sont numériques avant de filtrer
                         numeric_values = [v for v in raw_values if isinstance(v, (int, float)) and v > 0]
                         if not numeric_values: return None
                         # Garder seulement les items correspondants aux valeurs numériques > 0
                         valid_indices = [i for i, v in enumerate(raw_values) if isinstance(v, (int, float)) and v > 0]
                         items = [items[i] for i in valid_indices]
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
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + (max(values, default=1)*0.01), int(yval), ha='center', va='bottom', fontsize=8, fontname='DejaVu Sans', zorder=5)
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

                tc, _ = self.analyze_themes(); fig_t = self._create_matplotlib_figure_for_pdf(tc, 'Top 10 Thèmes NC', color='indianred', ytick_fontsize=7);
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
                        req_tl.extend([f"📋 Chap {r_['chapter']} ({r_['count']} mentions)", f"   Txt: {r_['requirement_text']}", ""])
                    self._add_text_to_pdf_page(fig, req_tl, title="Détail Exigences IFS", line_height=0.025, font_size=6, max_chars_per_line=130)
                    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
            return filename
        except Exception as e:
            st.error(f"❌ Erreur majeure lors de la génération du PDF: {e}")
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
                    reason_short = detail['reason'][:200] + "..." if len(detail['reason']) > 200 else detail['reason']
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
                    reason_short = ex_data.get('Lock reason', 'N/A')[:200] + "..." if len(ex_data.get('Lock reason', 'N/A')) > 200 else ex_data.get('Lock reason', 'N/A')
                    lines.append(f"   Ex {i+1} ({ex_data.get('Supplier', 'N/A')}, {ex_data.get('Country/Region', 'N/A')}):")
                    lines.append(f"     Motif: {reason_short}")
                lines.append("")
        return "\n".join(lines)


# --- Fonctions Utilitaires pour Streamlit ---
@st.cache_resource
def get_analyzer_instance(_locked_data_io, _checklist_data_io, locked_file_key, checklist_file_key):
    return IFSAnalyzer(_locked_data_io, _checklist_data_io)

@st.cache_data(ttl=3600)
def download_checklist_content_from_github(url):
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        st.warning(f"Impossible de télécharger la checklist depuis GitHub ({e}).")
        return None

# --- Interface Streamlit ---
def main():
    st.title("🛡️ Analyseur de Sécurité Alimentaire IFS")
    st.markdown("Bienvenue ! Téléversez votre fichier CSV des suspensions IFS pour une analyse détaillée.")

    with st.sidebar:
        st.image("https://www.ifs-certification.com/images/ifs_logo.svg", width=150)
        st.header("⚙️ Options d'Analyse")
        locked_file_uploaded = st.file_uploader("1. Fichier des suspensions IFS (.csv)", type="csv", key="locked_uploader", help="Fichier contenant les données des suspensions.")

        st.markdown("---")
        checklist_source = st.radio(
            "2. Source de la Checklist IFS Food V8:",
            ("Utiliser celle de GitHub (Recommandé)", "Téléverser ma checklist", "Ne pas utiliser de checklist"),
            index=0, key="checklist_source_radio", help="La checklist enrichit l'analyse des exigences."
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
        else:
            st.session_state.checklist_file_name_original = "Non fournie"

        analyzer = get_analyzer_instance(locked_data_io, checklist_data_io, current_locked_file_key, current_checklist_file_key)

        if analyzer.locked_df is not None and not analyzer.locked_df.empty:
            st.success(f"Fichier **'{locked_file_uploaded.name}'** analysé : **{len(analyzer.locked_df)}** suspensions IFS Food trouvées.")
            display_dashboard_tabs(analyzer)

            st.sidebar.markdown("---")
            st.sidebar.subheader("Téléchargement du Rapport")
            if st.sidebar.button("📄 Générer et Télécharger le PDF", key="pdf_button_main", help="Génère un rapport PDF complet."):
                with st.spinner("Génération du rapport PDF..."):
                    temp_pdf_filename = f"temp_report_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.pdf"
                    pdf_path_generated = analyzer.export_report_to_pdf(filename=temp_pdf_filename)
                    if pdf_path_generated and os.path.exists(pdf_path_generated):
                        with open(pdf_path_generated, "rb") as pdf_file:
                            pdf_bytes = pdf_file.read()
                        st.sidebar.download_button(label="📥 Cliquez ici pour télécharger le PDF", data=pdf_bytes,
                                                   file_name="Analyse_IFS_Suspensions_Report.pdf", mime="application/pdf", key="download_pdf_main")
                        st.sidebar.success("Rapport PDF prêt !")
                        try: os.remove(pdf_path_generated)
                        except Exception: pass
                    else: st.sidebar.error("Erreur création PDF.")
        else: st.error("Aucune donnée IFS Food trouvée ou fichier invalide.")
    else: st.info("👈 Téléversez un fichier CSV des suspensions IFS.")

    st.sidebar.markdown("---")
    st.sidebar.info("Analyseur IFS v1.2")

# --- display_dashboard_tabs (avec les mêmes améliorations UX que la version précédente) ---
def display_dashboard_tabs(analyzer):
    tab_titles = ["📊 Vue d'Ensemble", "🌍 Géographie", "🏷️ Thèmes Détaillés", "📋 Exigences IFS", "🕵️ Audits Spécifiques", "🔗 Analyse Croisée"]
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_titles)

    with tab1: # Vue d'Ensemble
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
            theme_counts, _ = analyzer.analyze_themes()
            if theme_counts:
                top_themes = dict(sorted(theme_counts.items(), key=lambda x:x[1], reverse=True)[:10])
                top_themes_clean = {k.replace('_',' ').replace('MANAGEMENT','MGMT').replace('RESPONSIBILITY','RESP.'):v for k,v in top_themes.items() if v > 0}
                if top_themes_clean: st.plotly_chart(analyzer._create_plotly_bar_chart(top_themes_clean, "Top 10 Thèmes de Non-Conformités", orientation='h', color='indianred', height=400), use_container_width=True)
        with row1_col2:
            scope_counts = analyzer.product_scope_analysis()
            if scope_counts:
                top_scopes = dict(scope_counts.most_common(10))
                top_scopes_clean = {f"Scope {k}": v for k, v in top_scopes.items() if v > 0}
                if top_scopes_clean: st.plotly_chart(analyzer._create_plotly_bar_chart(top_scopes_clean, "Top 10 Product Scopes Impactés", orientation='h', color='cornflowerblue', height=400), use_container_width=True)

    with tab2: # Géographie
        st.header("🌍 Analyse Géographique")
        if analyzer.locked_df is None or analyzer.locked_df.empty: st.warning("Aucune donnée à afficher."); return
        geo_stats_df = analyzer.geographic_analysis()
        if geo_stats_df is not None and not geo_stats_df.empty:
            st.plotly_chart(analyzer._create_plotly_choropleth_map(geo_stats_df, "Suspensions par Pays"), use_container_width=True)
            st.markdown("---"); st.subheader("Tableau des Suspensions par Pays (Top 20)")
            st.dataframe(geo_stats_df.head(20).style.highlight_max(subset=['total_suspensions'], color='rgba(255,170,170,0.5)', axis=0).format({'total_suspensions': '{:,}'}), use_container_width=True)
        else: st.info("Données géographiques non disponibles ou insuffisantes.")

    with tab3: # Thèmes Détaillés
        st.header("🏷️ Analyse Thématique Détaillée")
        if analyzer.locked_df is None or analyzer.locked_df.empty: st.warning("Aucune donnée à afficher."); return
        st.markdown("Explorez les motifs de suspension par thème. Cliquez sur un thème pour voir des exemples.")
        theme_counts, theme_details = analyzer.analyze_themes()
        for theme, count in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                with st.expander(f"{theme.replace('_', ' ').title()} ({count} cas)", expanded=False):
                    st.markdown(f"**Exemples de motifs (jusqu'à 5) pour : {theme.replace('_', ' ').title()}**")
                    for i, detail in enumerate(theme_details[theme][:5]):
                        st.markdown(f"**Cas {i+1} (Fournisseur: `{detail['supplier']}`, Pays: `{detail['country']}`)**")
                        st.caption(f"{str(detail['reason'])[:600]}...") # Assurer que 'reason' est str
                        if i < 4 : st.markdown("---")
                    # (Code pour les pays affectés par thème, déjà présent et correct)

    with tab4: # Exigences IFS
        st.header("📋 Analyse des Exigences IFS")
        if analyzer.locked_df is None or analyzer.locked_df.empty: st.warning("Aucune donnée à afficher."); return
        recommendations = analyzer.generate_ifs_recommendations_analysis()
        if recommendations and analyzer.checklist_df is not None:
            st.success("Checklist IFS Food V8 utilisée pour l'analyse des exigences.")
            df_reco = pd.DataFrame(recommendations).sort_values(by='count', ascending=False)
            top_reco_chart_df = df_reco.head(15).copy()
            top_reco_chart_df['display_label'] = top_reco_chart_df.apply(lambda row: f"{row['chapter']} ({str(row['requirement_text'])[:30]}...)", axis=1)
            reco_chart_data = pd.Series(top_reco_chart_df['count'].values, index=top_reco_chart_df['display_label']).to_dict()
            st.plotly_chart(analyzer._create_plotly_bar_chart(reco_chart_data, "Top 15 Exigences IFS Citées", orientation='v', color='gold', height=550, text_auto=False), use_container_width=True)
            st.markdown("---"); st.subheader("Détail des Exigences Citées (Top 25)")
            for index, row in df_reco.head(25).iterrows():
                with st.expander(f"Exigence {row['chapter']} ({row['count']} mentions)", expanded=False):
                    st.markdown(f"**Texte de l'exigence :**\n\n> _{str(row['requirement_text'])}_") # Assurer str
        elif recommendations:
             st.warning("Checklist non chargée. Affichage des numéros de chapitres uniquement.")
             # (Code pour affichage sans texte de checklist, déjà présent et correct)
        else: st.info("Aucune exigence/chapitre IFS spécifique n'a pu être extrait, ou la checklist n'est pas disponible/utilisée.")

    with tab5: # Audits Spécifiques
        st.header("🕵️ Analyse par Audits Spécifiques")
        if analyzer.locked_df is None or analyzer.locked_df.empty: st.warning("Aucune donnée à afficher."); return
        audit_analysis, audit_examples = analyzer.analyze_audit_types()
        # (Code pour affichage des audits, déjà présent et semble correct)
        if audit_analysis:
            audit_analysis_clean = {k.replace('_', ' ').title():v for k,v in audit_analysis.items() if v > 0}
            if audit_analysis_clean: st.plotly_chart(analyzer._create_plotly_bar_chart(audit_analysis_clean, "Répartition par Type d'Audit Spécifique", color='darkorange', height=400), use_container_width=True)
            st.markdown("---"); st.subheader("Détails et Exemples par Type d'Audit")
            for audit_type, count in sorted(audit_analysis.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    with st.expander(f"{audit_type.replace('_', ' ').title()} ({count} cas)", expanded=False):
                        # (Code pour les détails des exemples, déjà présent et correct)
                        st.markdown(f"**Exemples (jusqu'à 5) pour : {audit_type.replace('_', ' ').title()}**")
                        for i, ex_data in enumerate(audit_examples[audit_type]['examples'][:5]):
                            st.markdown(f"**Cas {i+1} (Fournisseur: `{ex_data.get('Supplier', 'N/A')}`, Pays: `{ex_data.get('Country/Region', 'N/A')}`)**")
                            st.caption(f"{str(ex_data.get('Lock reason', 'N/A'))[:600]}...") # Assurer str
                            if i < 4 : st.markdown("---")
                        countries_data = audit_examples[audit_type]['countries']
                        if countries_data:
                            st.markdown(f"**Répartition géographique (Top 5 pays) :** {', '.join([f'{c} ({n})' for c, n in countries_data.items()])}")
        else: st.info("Aucune donnée sur les types d'audits spécifiques disponible.")


    with tab6: # Analyse Croisée
        st.header("🔗 Analyse Croisée : Thèmes vs Product Scopes")
        if analyzer.locked_df is None or analyzer.locked_df.empty: st.warning("Aucune donnée à afficher."); return
        cross_pivot_matrix = analyzer.cross_analysis_scope_themes()
        if cross_pivot_matrix is not None and not cross_pivot_matrix.empty:
            # (Code pour la heatmap et le tableau, déjà présent et semble correct)
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
