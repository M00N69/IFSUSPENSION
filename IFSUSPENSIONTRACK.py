import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Mode non-interactif pour matplotlib, crucial pour serveurs/cloud
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
# import traceback # Pour débogage

# --- Configuration de la Page Streamlit ---
st.set_page_config(
    page_title="Analyseur Sécurité Alimentaire IFS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Fonction pour charger le CSS externe ---
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Fichier CSS '{file_name}' non trouvé. Les styles par défaut de Streamlit seront utilisés.")

# --- Classe IFSAnalyzer ---
class IFSAnalyzer:
    def __init__(self, locked_file_io, checklist_file_io=None):
        self.locked_df = None
        self.checklist_df = None
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
                'text': ['integrity program audit', 'ifs integrity check', 'on-site check ioc', 'unannounced audit issue', 'during the ioc audit', 'during ifs on site integrity check audit'],
                'chapters': []
            }
        }
        self.country_name_mapping = {
            "Allemagne": "Germany", "Italie": "Italy", "Pays-Bas": "Netherlands",
            "Espagne": "Spain", "Pologne": "Poland", "France": "France",
            "Belgique": "Belgium", "Autriche": "Austria", "Grèce": "Greece",
            "Turquie": "Turkey", "Danemark": "Denmark", "Royaume-Uni": "United Kingdom", "UK": "United Kingdom",
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
        }
        self.load_data(locked_file_io, checklist_file_io)
        if self.locked_df is not None:
            self.clean_lock_reasons()
            if 'Lock reason' in self.locked_df.columns:
                 self.locked_df['extracted_chapters_for_theme_analysis'] = self.locked_df['Lock reason'].apply(
                    lambda x: self.extract_ifs_chapters(x) if pd.notna(x) else []
                )
            else:
                self.locked_df['extracted_chapters_for_theme_analysis'] = pd.Series([[] for _ in range(len(self.locked_df))], index=self.locked_df.index)

    def load_data(self, locked_file_io, checklist_file_io=None):
        try:
            self.locked_df_original = pd.read_csv(locked_file_io, encoding='utf-8')
            if 'Standard' in self.locked_df_original.columns:
                self.locked_df = self.locked_df_original[
                    self.locked_df_original['Standard'].astype(str).str.contains('IFS Food', case=True, na=False)
                ].copy()
                if self.locked_df.empty:
                    st.warning("Aucune entrée 'IFS Food' trouvée après filtrage.")
                    self.locked_df = None; return
            else:
                st.warning("Colonne 'Standard' non trouvée. Analyse sur toutes les données.")
                self.locked_df = self.locked_df_original.copy()

            if self.locked_df is None : return

            if checklist_file_io:
                try:
                    temp_checklist_df = pd.read_csv(checklist_file_io, encoding='utf-8')
                    target_num_col = 'Requirement Number'; target_text_col = 'Requirement text (English)'
                    possible_num_cols = ['NUM_REQ', 'Requirement Number', 'Requirement No.', 'Exigence N°']
                    possible_text_cols = ['IFS Requirements', 'Requirement text (English)', 'Texte Exigence (Anglais)', 'Texte Exigence']
                    actual_num_col = next((col for col in possible_num_cols if col in temp_checklist_df.columns), None)
                    actual_text_col = next((col for col in possible_text_cols if col in temp_checklist_df.columns), None)

                    if actual_num_col and actual_text_col:
                        self.checklist_df = temp_checklist_df.rename(columns={actual_num_col: target_num_col, actual_text_col: target_text_col})
                    else:
                        missing = []
                        if not actual_num_col: missing.append("colonne des numéros d'exigence (ex: NUM_REQ)")
                        if not actual_text_col: missing.append("colonne des textes d'exigence (ex: IFS Requirements)")
                        st.warning(f"Colonnes requises pour la checklist ({', '.join(missing)}) non trouvées.")
                        self.checklist_df = None
                except Exception as e: st.error(f"Erreur chargement checklist: {e}"); self.checklist_df = None
        except Exception as e: st.error(f"❌ Erreur critique chargement suspensions: {e}"); self.locked_df = None

    def clean_lock_reasons(self):
        if self.locked_df is None or 'Lock reason' not in self.locked_df.columns: return
        self.locked_df['lock_reason_clean'] = self.locked_df['Lock reason'].astype(str).fillna('') \
            .str.lower().str.replace(r'[\n\r\t]', ' ', regex=True) \
            .str.replace(r'[^\w\s\.\-\/\%§]', ' ', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()

    def extract_ifs_chapters(self, text):
        if pd.isna(text) or not isinstance(text, str) or text.strip() == '': return []
        patterns = [
            r'(?:ko|major|cl\.|req\.|requirement(?: item)?|chapter|section|point|§|cl\s+|clause)?\s*(\d\.\d{1,2}(?:\.\d{1,2})?)(?!\s*\d{2,4})',
            r'(\d\.\d{1,2}(?:\.\d{1,2})?)\s*(?:ko|major|:|-|\(ko\)|\(major\))(?!\s*\d{2,4})',
            r'(?<!\d\.)(\d)\s*-\s*ko', r'requirement\s+(\d\.\d\.\d)(?!\s*\d{2,4})',
            r'cl\s+(\d\.\d+(?:\.\d+)?)(?!\s*\d{2,4})', r'§\s*(\d\.\d+(?:\.\d+)?)(?!\s*\d{2,4})',
            r'point\s+(\d\.\d+(?:\.\d+)?)(?!\s*\d{2,4})', r'item\s+(\d\.\d+(?:\.\d+)?)(?!\s*\d{2,4})']
        cf = []
        nt = text.lower().replace('\n', ' ').replace('\r', ' ')
        for p in patterns:
            for m in re.findall(p, nt):
                cnm = m if isinstance(m, str) else (m[-1] if isinstance(m, tuple) and m[-1] else m[0] if isinstance(m, tuple) and m[0] else None)
                if cnm:
                    cn = str(cnm).strip().rstrip('.').strip()
                    if re.fullmatch(r'\d(\.\d+){0,2}', cn):
                        mp = cn.split('.')[0]
                        if mp.isdigit() and 1 <= int(mp) <= 6: cf.append(cn)
        return sorted(list(set(cf)))

    def analyze_themes(self):
        if self.locked_df is None or 'lock_reason_clean' not in self.locked_df.columns: return {}, {}
        ta = []
        if 'extracted_chapters_for_theme_analysis' not in self.locked_df.columns:
            if 'Lock reason' in self.locked_df.columns:
                 self.locked_df['extracted_chapters_for_theme_analysis'] = self.locked_df['Lock reason'].apply(
                    lambda x: self.extract_ifs_chapters(x) if pd.notna(x) else [])
            else: return {}, {}

        for idx, row in self.locked_df.iterrows():
            rc = row.get('lock_reason_clean', ''); orig_r = row.get('Lock reason', '')
            s = row.get('Supplier', 'N/A'); c = row.get('Country/Region', 'N/A')
            ec = row['extracted_chapters_for_theme_analysis']; bt = 'NON_CLASSIFIE'; ms = 0
            admin_tn = 'ADMINISTRATIVE_OPERATIONAL_ISSUES'
            admin_td = self.themes_definition.get(admin_tn, {}); admin_kws = admin_td.get('text', [])
            is_admin = any(re.search(r'\b'+re.escape(kw.lower())+r'\b',rc) for kw in admin_kws)
            if is_admin: bt=admin_tn; ms=200
            else:
                for tn, td in self.themes_definition.items():
                    if tn == admin_tn: continue
                    cs_ = sum(100 for chk in td.get('chapters',[]) if chk in ec)
                    tms = 0
                    for kw_ in td.get('text',[]):
                        if re.search(r'\b'+re.escape(kw_.lower())+r'\b',rc): tms+=20
                        elif kw_.lower() in rc: tms+=5
                    cs_+=tms
                    if cs_>ms: ms=cs_; bt=tn
            if bt!=admin_tn and ms<15: bt='NON_CLASSIFIE'
            ta.append({'theme':bt,'reason':orig_r,'supplier':s,'country':c})
        fcc=Counter(t['theme'] for t in ta); fcd={tn:[] for tn in list(self.themes_definition.keys())+['NON_CLASSIFIE']}
        for t_assign in ta: fcd[t_assign['theme']].append({"reason":t_assign['reason'],"supplier":t_assign['supplier'],"country":t_assign['country']})
        return fcc,fcd
    
    def get_reasons_for_chapter(self, chapter_number):
        if self.locked_df is None or 'Lock reason' not in self.locked_df.columns or 'extracted_chapters_for_theme_analysis' not in self.locked_df.columns: return []
        rl = []
        for idx, row in self.locked_df.iterrows():
            if chapter_number in row['extracted_chapters_for_theme_analysis']:
                rl.append({"supplier": row.get('Supplier', 'N/A'), "country": row.get('Country/Region', 'N/A'), "reason_text": row.get('Lock reason', '')})
        return rl

    def geographic_analysis(self):
        if self.locked_df is None or 'Country/Region' not in self.locked_df.columns: return None
        geo_df = self.locked_df.groupby('Country/Region').size().sort_values(ascending=False).reset_index(name='total_suspensions')
        geo_df['Country/Region_EN'] = geo_df['Country/Region'].map(self.country_name_mapping).fillna(geo_df['Country/Region'])
        return geo_df

    def clean_product_scopes(self, scope_text):
        if pd.isna(scope_text): return []
        scope_text = str(scope_text)
        raw_scopes = re.split(r'[,;\s"\'’`]|et|\/|&|\.\s', scope_text)
        cleaned_scopes = []
        for scope in raw_scopes:
            scope = scope.strip().replace('"', '').replace("'", "")
            if not scope or not scope.isdigit(): continue
            num = int(scope)
            if 1 <= num <= 11: cleaned_scopes.append(str(num))
            elif num > 1000:
                ps2 = str(num % 100); ps1 = str(num % 10)
                if ps2 in ['10', '11']: cleaned_scopes.append(ps2)
                elif ps1 in [str(i) for i in range(1,10)]: cleaned_scopes.append(ps1)
        return list(set(cleaned_scopes))

    def product_scope_analysis(self):
        if self.locked_df is None or 'Product scopes' not in self.locked_df.columns: return None
        all_s = []; [all_s.extend(self.clean_product_scopes(st)) for st in self.locked_df['Product scopes'].dropna()]
        return Counter(all_s)

    def chapter_frequency_analysis(self):
        if self.locked_df is None or 'Lock reason' not in self.locked_df.columns: return Counter()
        all_c = []; [all_c.extend(self.extract_ifs_chapters(r)) for r in self.locked_df['Lock reason'].dropna()]
        return Counter(all_c)

    def analyze_audit_types(self):
        if self.locked_df is None: return {}, {}
        audit_keywords_definition = {
            'INTEGRITY_PROGRAM_IP': ['integrity program', 'integrity', 'programme intégrité', 'programme integrity','onsite check', 'on site check', 'on-site check', 'on-site integrity check', 'ioc', 'i.o.c', 'ip audit', 'integrity audit', 'spot check', 'unannounced audit', 'audit inopiné', 'control inopiné', 'ifs integrity', 'during the ioc audit', 'during ifs on site integrity check audit'],
            'SURVEILLANCE_FOLLOW_UP': ['surveillance', 'surveillance audit', 'follow up audit', 'follow-up', 'suivi', 'corrective action'],
            'COMPLAINT_WITHDRAWAL': ['complaint', 'réclamation', 'plainte', 'customer complaint', 'withdrawal', 'retrait', 'recall'],
            'RECERTIFICATION_RENEWAL': ['recertification', 'renewal', 'renouvellement', 're-certification', 'renewal audit']
        }
        aa = {at: 0 for at in audit_keywords_definition}; ae = {at: {'examples': [], 'countries': Counter()} for at in audit_keywords_definition}
        for idx, row in self.locked_df.iterrows():
            txt = (str(row.get('Lock reason', '')) + " " + str(row.get('Lock history', ''))).lower()
            for at, kws in audit_keywords_definition.items():
                if any(keyword.lower() in txt for keyword in kws):
                    aa[at] += 1
                    # Garder TOUS les exemples pour IP
                    if at == 'INTEGRITY_PROGRAM_IP' or len(ae[at]['examples']) < 5 : # Limiter les autres à 5
                        ae[at]['examples'].append({'Supplier': row.get('Supplier', 'N/A'), 
                                                   'Country/Region': row.get('Country/Region', 'N/A'), 
                                                   'Lock reason': row.get('Lock reason', 'N/A')})
                    ae[at]['countries'][row.get('Country/Region', 'N/A')] += 1
        for at in ae: ae[at]['countries'] = dict(ae[at]['countries'].most_common(5))
        return aa, ae

    def generate_ifs_recommendations_analysis(self):
        if self.locked_df is None or self.checklist_df is None or 'Requirement Number' not in self.checklist_df.columns or 'Requirement text (English)' not in self.checklist_df.columns: return None
        cc = self.chapter_frequency_analysis();
        if not cc: return None
        recs = []
        for ch, cnt in cc.most_common():
            nc = ch.replace("KO ", "").strip()
            mask = self.checklist_df['Requirement Number'].astype(str).str.strip() == nc.strip()
            rts = self.checklist_df.loc[mask, 'Requirement text (English)']
            rt = rts.iloc[0] if not rts.empty else f"Texte non trouvé pour '{nc}'."
            srs = self.get_reasons_for_chapter(ch)
            recs.append({'chapter': ch, 'count': cnt, 'requirement_text': rt, 'specific_reasons': srs})
        return recs

    def cross_analysis_scope_themes(self):
        if self.locked_df is None or 'Product scopes' not in self.locked_df.columns or 'lock_reason_clean' not in self.locked_df.columns: return None
        technical_themes_for_cross = ['HYGIENE_PERSONNEL', 'HACCP_CCP_OPRP', 'TRACEABILITY', 'ALLERGEN_MANAGEMENT', 'PEST_CONTROL', 'CLEANING_SANITATION', 'MAINTENANCE_EQUIPMENT_INFRASTRUCTURE', 'FOREIGN_BODY_CONTAMINATION', 'LABELLING_PRODUCT_INFORMATION', 'QUANTITY_CONTROL_WEIGHT_MEASUREMENT']
        scope_theme_data = []
        if 'extracted_chapters_for_theme_analysis' not in self.locked_df.columns: return None

        for idx, row in self.locked_df.iterrows():
            scopes_text, reason_text_clean = row['Product scopes'], row['lock_reason_clean']
            extracted_chaps_for_row = row.get('extracted_chapters_for_theme_analysis', [])
            if pd.notna(scopes_text) and pd.notna(reason_text_clean) and reason_text_clean:
                for scope in self.clean_product_scopes(scopes_text):
                    for theme_key in technical_themes_for_cross:
                        theme_data = self.themes_definition.get(theme_key, {})
                        keywords = theme_data.get('text', [])
                        chapters_kw = theme_data.get('chapters', [])
                        theme_matched = any(kw.lower() in reason_text_clean for kw in keywords) or \
                                        any(chap in extracted_chaps_for_row for chap in chapters_kw)
                        if theme_matched:
                            scope_theme_data.append({'scope': f"Scope {scope}", 'theme': theme_key.replace("_", " ").title()[:20]})
        if not scope_theme_data: return None
        dfc = pd.DataFrame(scope_theme_data) # dfc est défini ici
        return None if dfc.empty else dfc.pivot_table(index='scope', columns='theme', aggfunc='size', fill_value=0) # Correction ici

    def _create_plotly_bar_chart(self, data_dict, title, orientation='v', xaxis_title="", yaxis_title="", color='royalblue', height=400, text_auto=True):
        if not data_dict : return go.Figure()
        filtered_data_dict = {k: v for k, v in data_dict.items() if isinstance(v, (int, float)) and v > 0}
        if not filtered_data_dict: return go.Figure()
        if orientation == 'h': sorted_data = dict(sorted(filtered_data_dict.items(), key=lambda item: item[1]))
        else: sorted_data = dict(sorted(filtered_data_dict.items(), key=lambda item: item[1], reverse=True))
        y_data_plot, x_data_plot = (list(sorted_data.keys()), list(sorted_data.values())) if orientation == 'h' else (list(sorted_data.values()), list(sorted_data.keys()))
        text_on_bars = [val if val > 0 else '' for val in (x_data_plot if orientation == 'h' else y_data_plot)]
        fig = go.Figure(go.Bar(x=x_data_plot, y=y_data_plot, orientation=orientation, marker_color=color, text=text_on_bars, textposition='outside' if text_auto else None, textfont_size=9))
        fig.update_layout(title={'text': f"<b>{title}</b>", 'x':0.5, 'font': {'size': 16, 'family': "Arial, sans-serif"}}, xaxis_title=xaxis_title, yaxis_title=yaxis_title, height=height, margin=dict(l=10, r=10, t=60, b=40), font=dict(family="Arial, sans-serif", size=10), yaxis=dict(tickfont_size=9) if orientation == 'h' else dict(tickfont_size=9, autorange="reversed" if orientation=='v' and len(y_data_plot)>10 else None), xaxis=dict(tickfont_size=9), plot_bgcolor='var(--card-background)', paper_bgcolor='var(--card-background)', font_color='var(--text-color)')
        if orientation == 'v': fig.update_xaxes(categoryorder='total descending')
        return fig

    def _create_plotly_choropleth_map(self, geo_data_df, title, height=500):
        if geo_data_df is None or geo_data_df.empty or 'Country/Region_EN' not in geo_data_df.columns: return go.Figure()
        fig = px.choropleth(geo_data_df, locations="Country/Region_EN", locationmode='country names', color="total_suspensions", hover_name="Country/Region", color_continuous_scale=px.colors.sequential.Blues, title=title, height=height)
        fig.update_layout(title={'text': f"<b>{title}</b>", 'x':0.5, 'font': {'size': 16}}, geo=dict(showframe=False, showcoastlines=True, projection_type='natural earth', bgcolor='rgba(235,245,255,0.1)', landcolor='rgba(217, 217, 217, 0.5)'), margin=dict(l=0, r=0, t=50, b=0), paper_bgcolor='var(--card-background)', font_color='var(--text-color)', coloraxis_colorbar=dict(title="Suspensions", tickfont_color='var(--text-color-muted)'))
        return fig

    def _create_plotly_heatmap(self, pivot_matrix, title, height=500):
        if pivot_matrix is None or pivot_matrix.empty: return go.Figure()
        fig = px.imshow(pivot_matrix, text_auto='.0f', aspect="auto", color_continuous_scale='Blues', title=title, height=height)
        fig.update_layout(title={'text': f"<b>{title}</b>", 'x':0.5, 'font': {'size': 16}}, margin=dict(l=10, r=10, t=80, b=10), xaxis=dict(tickangle=35, side='bottom', tickfont_size=9), yaxis=dict(tickfont_size=9), paper_bgcolor='var(--card-background)', plot_bgcolor='var(--card-background)', font_color='var(--text-color)')
        fig.update_traces(hovertemplate="Scope: %{y}<br>Thème: %{x}<br>Cas: %{z}<extra></extra>")
        return fig

    def _add_text_to_pdf_page(self, fig, text_lines, start_y=0.95, line_height=0.035, font_size=9, title="", title_font_size=14, max_chars_per_line=100):
        ax = fig.gca(); ax.clear(); ax.axis('off')
        if title: ax.text(0.5, start_y, title, ha='center', va='top', fontsize=title_font_size, fontweight='bold', fontname='DejaVu Sans'); start_y -= (line_height * 2.5)
        cy = start_y
        for line in text_lines:
            import textwrap
            wls = textwrap.wrap(line, width=max_chars_per_line, break_long_words=False, replace_whitespace=False)
            for wl in wls:
                if cy < 0.05: return False
                fw = 'bold' if line.startswith(tuple(["🎯","📊","🌍","🏭","📋","🔍", "---"])) else 'normal'
                fs = font_size + 1 if fw == 'bold' else font_size
                if line.startswith("---"): fs -=1
                ax.text(0.03, cy, wl, ha='left', va='top', fontsize=fs, fontweight=fw, fontname='DejaVu Sans'); cy -= line_height
            if not line.strip(): cy -= (line_height * 0.3)
        return True

    def _create_matplotlib_figure_for_pdf(self, data_dict_or_df, title, x_label="", y_label="", chart_type='barh', top_n=10, color='skyblue', xtick_rotation=0, ytick_fontsize=8):
        if not data_dict_or_df and not isinstance(data_dict_or_df, pd.DataFrame) : return None
        fig, ax = plt.subplots(figsize=(10, 6.5)); items, values = [], []
        if isinstance(data_dict_or_df, (Counter, dict)):
            fd = {k: v for k, v in data_dict_or_df.items() if isinstance(v, (int, float)) and v > 0};
            if not fd: return None
            sd = dict(sorted(fd.items(), key=lambda item: item[1], reverse=(chart_type=='bar'))[:top_n])
            items = [str(k).replace('_',' ').replace('MANAGEMENT','MGMT').replace('RESPONSIBILITY','RESP.')[:35] for k in sd.keys()]; values = list(sd.values())
        elif isinstance(data_dict_or_df, pd.DataFrame):
            dft = data_dict_or_df.head(top_n)
            if 'Country/Region' in dft.columns and 'total_suspensions' in dft.columns: items=dft['Country/Region'].tolist(); values=dft['total_suspensions'].tolist(); chart_type='bar'
            elif 'chapter' in dft.columns and 'count' in dft.columns and 'requirement_text' in dft.columns:
                 dftf = dft[dft['count'] > 0];
                 if dftf.empty: return None
                 items = [f"{r['chapter']}\n({str(r['requirement_text'])[:40]}...)" if r['requirement_text'] != "Texte de l'exigence non trouvé dans la checklist fournie." else r['chapter'] for i, r in dftf.iterrows()]
                 values = dftf['count'].tolist(); chart_type = 'bar'
            else:
                if not dft.empty and len(dft.columns) > 0 :
                    fc = dft.columns[0]; items = dft[fc].astype(str).tolist()
                    if len(dft.columns) > 1: sc = dft.columns[1]; rvs = dft[sc].tolist()
                    else: items = dft.index.astype(str).tolist(); rvs = dft[fc].tolist()
                    nvs = [v for v in rvs if isinstance(v, (int, float)) and v > 0];
                    if not nvs: return None
                    vis = [i for i, v in enumerate(rvs) if isinstance(v, (int, float)) and v > 0]
                    items = [items[i] for i in vis][:len(nvs)]; values = nvs
                else: return None
        if not items or not values or all(v == 0 for v in values): return None
        if chart_type == 'barh':
            yp = np.arange(len(items)); ax.barh(yp, values, color=color, edgecolor='grey', zorder=3)
            ax.set_yticks(yp); ax.set_yticklabels(items, fontsize=ytick_fontsize, fontname='DejaVu Sans'); ax.invert_yaxis()
            ax.set_xlabel(x_label or 'Nombre de cas', fontsize=10, fontname='DejaVu Sans')
            for i, v_ in enumerate(values): ax.text(v_+(max(values,default=1)*0.01), i, str(v_), va='center', fontsize=8, fontname='DejaVu Sans', zorder=5)
            ax.set_xlim(0, max(values, default=1)*1.15)
        elif chart_type == 'bar':
            xp = np.arange(len(items)); bars = ax.bar(xp, values, color=color, edgecolor='grey', zorder=3, width=0.7)
            ax.set_xticks(xp); ax.set_xticklabels(items, rotation=xtick_rotation, ha='right' if xtick_rotation > 0 else 'center', fontsize=ytick_fontsize, fontname='DejaVu Sans')
            ax.set_ylabel(y_label or 'Nombre de cas', fontsize=10, fontname='DejaVu Sans')
            for bar in bars: yv = bar.get_height(); ax.text(bar.get_x()+bar.get_width()/2., yv+(max(values,default=1)*0.01), int(yv), ha='center', va='bottom', fontsize=8, fontname='DejaVu Sans', zorder=5)
            ax.set_ylim(0, max(values, default=1)*1.15)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=20, fontname='DejaVu Sans')
        ax.grid(axis='x' if chart_type == 'barh' else 'y', linestyle=':', alpha=0.6, zorder=0)
        sns.despine(left=True, bottom=True); plt.tight_layout(pad=2.0)
        return fig

    def export_report_to_pdf(self, filename='IFS_Analysis_Report.pdf'):
        if self.locked_df is None: return None
        try:
            with PdfPages(filename) as pdf:
                ts = len(self.locked_df)
                if ts == 0: fig=plt.figure(figsize=(8.5,11)); self._add_text_to_pdf_page(fig,["Aucune donnée."],title="Rapport"); pdf.savefig(fig); plt.close(fig); return filename
                fig=plt.figure(figsize=(8.5,11)); lno=st.session_state.get('locked_file_name_original','N/A'); cno=st.session_state.get('checklist_file_name_original','Non fournie')
                tt=[f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}", "", f"F Suspensions: {lno}", f"F Checklist: {cno}", "", "📊 VUE D'ENSEMBLE", f"   • Total: {ts}"]
                wrc=self.locked_df['Lock reason'].notna().sum(); tt.append(f"   • Avec motifs: {wrc} ({wrc/ts*100:.1f}% si ts>0 else 0%)")
                ass, _ = self.analyze_audit_types(); tas=sum(ass.values()); tt.append(f"   • Audits spécifiques: {tas} ({tas/ts*100:.1f}% si ts>0 else 0%)")
                self._add_text_to_pdf_page(fig,tt,title="Rapport Analyse IFS Food Safety"); pdf.savefig(fig,bbox_inches='tight'); plt.close(fig)
                tcf,_=self.analyze_themes(); tct={k:v for k,v in tcf.items() if k not in ['ADMINISTRATIVE_OPERATIONAL_ISSUES','NON_CLASSIFIE']}
                figt=self._create_matplotlib_figure_for_pdf(tct,'Top 10 Thèmes Tech NC','indianred',ytick_fontsize=7);
                if figt: pdf.savefig(figt,bbox_inches='tight'); plt.close(figt)
                gs=self.geographic_analysis(); figg=self._create_matplotlib_figure_for_pdf(gs,'Top 10 Pays','bar','lightseagreen',xtick_rotation=35,ytick_fontsize=7);
                if figg: pdf.savefig(figg,bbox_inches='tight'); plt.close(figg)
                sc=self.product_scope_analysis(); scp={f"Sc {k}":v for k,v in sc.items()}; figs=self._create_matplotlib_figure_for_pdf(scp,'Top 10 Product Scopes','cornflowerblue',ytick_fontsize=7);
                if figs: pdf.savefig(figs,bbox_inches='tight'); plt.close(figs)
                reco=self.generate_ifs_recommendations_analysis()
                if reco: dfr=pd.DataFrame(reco); figc=self._create_matplotlib_figure_for_pdf(dfr,'Top 10 Exigences IFS','bar','gold',xtick_rotation=35,ytick_fontsize=6)
                else: ccd=self.chapter_frequency_analysis(); figc=self._create_matplotlib_figure_for_pdf(ccd,'Top 10 Chapitres (Num)','bar','gold',xtick_rotation=35,ytick_fontsize=7)
                if figc: pdf.savefig(figc,bbox_inches='tight'); plt.close(figc)
                cpm=self.cross_analysis_scope_themes()
                if cpm is not None and not cpm.empty:
                    tn=min(10,len(cpm.index)); stot=cpm.sum(axis=1).sort_values(ascending=False)
                    cpmf=cpm.loc[stot.head(tn).index] if len(cpm.index)>tn else cpm
                    if not cpmf.empty and cpmf.shape[0]>0 and cpmf.shape[1]>0:
                        figh,axh=plt.subplots(figsize=(10,max(5,len(cpmf.index)*0.7)))
                        sns.heatmap(cpmf,annot=True,cmap="Blues",fmt='d',ax=axh,annot_kws={"size":7},linewidths=.5,linecolor='grey');
                        axh.set_title('Corrélations: Thèmes vs Scopes (Top)',fontsize=13,fontweight='bold',pad=20,fontname='DejaVu Sans')
                        axh.tick_params(axis='x',labelsize=8,rotation=35,ha='right'); axh.tick_params(axis='y',labelsize=8,rotation=0)
                        plt.tight_layout(pad=2.0); pdf.savefig(figh,bbox_inches='tight'); plt.close(figh)
                for gf,tit,lh,fs,mcpl in [(self.generate_detailed_theme_analysis_text,"Analyse Thématique Détaillée",0.03,8,110),(self.generate_audit_analysis_report_text,"Analyse Types d'Audits",0.03,8,110)]:
                    fig=plt.figure(figsize=(8.5,11)); txtc=gf(); self._add_text_to_pdf_page(fig,txtc.splitlines(),title=tit,line_height=lh,font_size=fs,max_chars_per_line=mcpl); pdf.savefig(fig,bbox_inches='tight'); plt.close(fig)
                if reco:
                    fig=plt.figure(figsize=(8.5,11)); rtl=["Note: Texte exigence checklist IFS Food v8 (si fournie).\n"]
                    for r_ in sorted(reco,key=lambda x:x['count'],reverse=True)[:25]: rtl.extend([f"📋 Chap {r_['chapter']} ({r_['count']} mentions)",f"   Txt: {str(r_['requirement_text'])}",""])
                    self._add_text_to_pdf_page(fig,rtl,title="Détail Exigences IFS",line_height=0.025,font_size=6,max_chars_per_line=130); pdf.savefig(fig,bbox_inches='tight'); plt.close(fig)
            return filename
        except Exception as e: st.error(f"❌ Erreur PDF Gen: {e}"); return None

    def generate_detailed_theme_analysis_text(self):
        if self.locked_df is None: return ""
        tc, td = self.analyze_themes(); lines=[]
        tct={k:v for k,v in tc.items() if k not in ['ADMINISTRATIVE_OPERATIONAL_ISSUES','NON_CLASSIFIE']}
        lines.append("--- THÈMES TECHNIQUES DE NON-CONFORMITÉ ---")
        for th, ct in sorted(tct.items(),key=lambda x:x[1],reverse=True):
            if ct>0:
                lines.append(f"\n🎯 {th.replace('_',' ').title()} ({ct} cas)"); lines.append("-"*60)
                for i,det in enumerate(td.get(th,[])[:3]): rs=str(det.get('reason','N/A'))[:200]+"..."; lines.append(f"   Ex {i+1} ({det.get('supplier','N/A')}, {det.get('country','N/A')}):\n     Motif: {rs}")
                lines.append("")
        aict=tc.get('ADMINISTRATIVE_OPERATIONAL_ISSUES',0)
        if aict>0:
            lines.append("\n--- PROBLÈMES ADMINISTRATIFS / OPÉRATIONNELS ---"); lines.append(f"\n🎯 Admin / Op ({aict} cas)"); lines.append("-"*60)
            for i,det in enumerate(td.get('ADMINISTRATIVE_OPERATIONAL_ISSUES',[])[:3]): rs=str(det.get('reason','N/A'))[:200]+"..."; lines.append(f"   Ex {i+1} ({det.get('supplier','N/A')}, {det.get('country','N/A')}): {rs}")
            lines.append("")
        uct=tc.get('NON_CLASSIFIE',0)
        if uct>0:
            lines.append("\n--- MOTIFS NON CLASSIFIÉS ---"); lines.append(f"\n🎯 Non Classifiés ({uct} cas)"); lines.append("-"*60)
            for i,det in enumerate(td.get('NON_CLASSIFIE',[])[:3]): rs=str(det.get('reason','N/A'))[:200]+"..."; lines.append(f"   Ex {i+1} ({det.get('supplier','N/A')}, {det.get('country','N/A')}): {rs}")
            lines.append("")
        return "\n".join(lines)

    def generate_audit_analysis_report_text(self):
        if self.locked_df is None: return ""
        aa,ae=self.analyze_audit_types(); ts=len(self.locked_df)
        if ts==0: return "Aucune suspension."
        lines=[f"Total audits spécifiques: {sum(aa.values())} ({sum(aa.values())/ts*100:.1f}% du total si ts>0 else 0%)"]
        for at,ct in sorted(aa.items(),key=lambda x:x[1],reverse=True):
            if ct>0:
                lines.append(f"\n🎯 {at.replace('_',' ').title()} ({ct} cas - {ct/ts*100:.1f}%)"); lines.append("-"*60)
                for i,exd in enumerate(ae[at]['examples'][:2]): rs=str(exd.get('Lock reason','N/A'))[:200]+"..."; lines.append(f"   Ex {i+1} ({exd.get('Supplier','N/A')}, {exd.get('Country/Region','N/A')}):\n     Motif: {rs}")
                lines.append("")
        return "\n".join(lines)

# >>> FIN DE LA CLASSE IFSAnalyzer <<<


# --- Fonctions Utilitaires pour Streamlit ---
@st.cache_resource(show_spinner="Chargement de l'analyseur...")
def get_analyzer_instance(_locked_data_io, _checklist_data_io, locked_file_key, checklist_file_key):
    # st.write(f"DEBUG Cache: locked='{locked_file_key}', checklist='{checklist_file_key}'")
    return IFSAnalyzer(_locked_data_io, _checklist_data_io)

@st.cache_data(ttl=3600, show_spinner="Téléchargement de la checklist IFS...")
def download_checklist_content_from_github(url):
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        st.error(f"Échec du téléchargement de la checklist depuis GitHub : {e}.")
        return None

# --- Interface Streamlit ---
def main():
    load_css("assets/styles.css")

    st.title("🛡️ Analyseur de Non-Conformités IFS")
    st.markdown("""
    **Analysez les suspensions de certificats IFS Food.** Téléversez votre fichier de données et explorez les tendances.
    La checklist IFS Food V8 est automatiquement téléchargée depuis GitHub pour une analyse détaillée des exigences.
    """)
    st.markdown("---")

    with st.sidebar:
        st.markdown("<div style='text-align: center; margin-bottom: 10px;'><img src='https://www.ifs-certification.com/images/ifs_logo.svg' width=180 alt='IFS Logo'></div>", unsafe_allow_html=True)
        st.header("Paramètres d'Analyse")
        locked_file_uploaded = st.file_uploader("1. Fichier Suspensions IFS (.csv)", type="csv", key="locked_uploader_main_v3", help="Sélectionnez le fichier CSV exporté de la base de données IFS.")
        st.session_state.checklist_file_name_original = "Checklist IFS Food V8 (GitHub)"

    if locked_file_uploaded is not None:
        with st.spinner("Préparation des données et de l'analyseur..."):
            current_locked_file_key = locked_file_uploaded.name + str(locked_file_uploaded.size)
            st.session_state.locked_file_name_original = locked_file_uploaded.name
            locked_data_io = io.BytesIO(locked_file_uploaded.getvalue())
            checklist_data_io = None
            checklist_url = "https://raw.githubusercontent.com/M00N69/Action-plan/main/Guide%20Checklist_IFS%20Food%20V%208%20-%20CHECKLIST.csv"
            checklist_text_content = download_checklist_content_from_github(checklist_url)
            current_checklist_file_key = f"gh_checklist_hash_{hash(checklist_text_content)}" if checklist_text_content else "gh_checklist_failed"
            if checklist_text_content: checklist_data_io = io.StringIO(checklist_text_content)
            
            analyzer = get_analyzer_instance(locked_data_io, checklist_data_io, current_locked_file_key, current_checklist_file_key)

        if analyzer.locked_df is not None and not analyzer.locked_df.empty:
            st.success(f"Fichier **'{locked_file_uploaded.name}'** analysé : **{len(analyzer.locked_df)}** suspensions IFS Food trouvées.")
            display_dashboard_tabs(analyzer)

            st.sidebar.markdown("---"); st.sidebar.subheader("Exporter le Rapport")
            if st.sidebar.button("📄 Générer et Télécharger le PDF", key="pdf_button_main_v3", help="Crée un rapport PDF complet.", type="primary"):
                with st.spinner("Génération du rapport PDF en cours..."):
                    temp_pdf_fn = f"temp_report_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.pdf"
                    pdf_path = analyzer.export_report_to_pdf(filename=temp_pdf_fn)
                    if pdf_path and os.path.exists(pdf_path):
                        with open(pdf_path, "rb") as f: pdf_bytes = f.read()
                        st.sidebar.download_button(label="📥 Télécharger le Rapport PDF", data=pdf_bytes, file_name="Rapport_Analyse_IFS.pdf", mime="application/pdf", key="dl_pdf_btn_v3")
                        st.sidebar.success("Rapport PDF prêt !")
                        try: os.remove(pdf_path)
                        except Exception: pass
                    else: st.sidebar.error("Erreur création PDF.")
        elif analyzer.locked_df is not None and analyzer.locked_df.empty:
             st.warning("Aucune suspension 'IFS Food' n'a été trouvée dans le fichier après filtrage.")
        else: st.error("Fichier suspensions non chargé correctement.")
    else:
        st.info("👈 Téléversez un fichier CSV des suspensions IFS.")

    st.sidebar.markdown("---"); st.sidebar.markdown("Analyseur IFS v1.6"); st.sidebar.markdown("Développé avec 💡 par IA")

def display_dashboard_tabs(analyzer):
    tab_titles = ["📊 Vue d'Ensemble", "🌍 Géographie & Audits", "🏷️ Thèmes Détaillés", "📋 Exigences IFS", "🕵️ Focus Audits IP", "🔗 Analyse Croisée"]
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_titles)

    with tab1:
        st.header("📊 Vue d'Ensemble des Suspensions")
        if analyzer.locked_df is None or analyzer.locked_df.empty: st.warning("Aucune donnée à afficher."); return
        total_suspensions = len(analyzer.locked_df)
        audit_analysis_summary, _ = analyzer.analyze_audit_types()
        ip_cases_count = audit_analysis_summary.get('INTEGRITY_PROGRAM_IP', 0)
        unique_countries_count = analyzer.locked_df['Country/Region'].nunique() if 'Country/Region' in analyzer.locked_df.columns else 0

        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Total Suspensions IFS Food", total_suspensions, help="Nombre total de suspensions après filtrage pour 'IFS Food'.")
        with col2: st.metric("Cas 'Integrity Program'", f"{ip_cases_count} ({ip_cases_count/total_suspensions*100:.1f}% du total)" if total_suspensions > 0 else f"{ip_cases_count}", help="Nombre de suspensions liées à des audits Integrity Program (IOC, On-site Check, etc.).")
        with col3: st.metric("Nombre de Pays Touchés", unique_countries_count, help="Nombre de pays uniques d'où proviennent les suspensions.")
        
        st.markdown("---"); st.subheader("Visualisations Clés des Non-Conformités Techniques")
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            theme_counts_full, _ = analyzer.analyze_themes()
            theme_counts_technical = {k: v for k, v in theme_counts_full.items() if k not in ['ADMINISTRATIVE_OPERATIONAL_ISSUES', 'NON_CLASSIFIE']}
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
        st.header("🌍 Analyse Géographique & Types d'Audit par Pays")
        if analyzer.locked_df is None or analyzer.locked_df.empty: st.warning("Aucune donnée géographique."); return
        geo_stats_df = analyzer.geographic_analysis()
        if geo_stats_df is not None and not geo_stats_df.empty:
            geo_stats_df_filtered = geo_stats_df[geo_stats_df['total_suspensions'] > 0]
            if not geo_stats_df_filtered.empty:
                top_n_countries_pie = 7
                pie_data_df = geo_stats_df_filtered.copy()
                if len(pie_data_df) > top_n_countries_pie:
                    other_sum = pie_data_df.iloc[top_n_countries_pie:]['total_suspensions'].sum()
                    pie_data_df = pie_data_df.head(top_n_countries_pie)
                    others_row = pd.DataFrame([{'Country/Region': 'Autres Pays', 'total_suspensions': other_sum, 'Country/Region_EN': 'Autres Pays'}])
                    pie_data_df = pd.concat([pie_data_df, others_row], ignore_index=True)
                
                fig_pie = px.pie(pie_data_df, values='total_suspensions', names='Country/Region', title=f'Distribution des Suspensions (Top {top_n_countries_pie} & Autres)', hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2)
                fig_pie.update_traces(textposition='inside', textinfo='percent+label', pull=[0.03] * len(pie_data_df))
                fig_pie.update_layout(legend_title_text='Pays', height=500, title_x=0.5, paper_bgcolor='var(--card-background)', plot_bgcolor='var(--card-background)', font_color='var(--text-color)')
                st.plotly_chart(fig_pie, use_container_width=True)
                st.markdown("---")

                st.subheader("Répartition des Types d'Audit pour les 10 Principaux Pays")
                top_10_countries_list = geo_stats_df_filtered['Country/Region'].head(10).tolist()
                country_audit_data = []
                audit_keywords_ip = analyzer.themes_definition.get('INTEGRITY_PROGRAM_SPECIFIC_ISSUES', {}).get('text', [])
                
                df_for_country_audit = analyzer.locked_df[analyzer.locked_df['Country/Region'].isin(top_10_countries_list)].copy() # Travailler sur une copie

                def get_audit_category(reason_history_text):
                    if any(kw.lower() in reason_history_text for kw in audit_keywords_ip): return 'Integrity Program'
                    # Ajoutez d'autres catégories si besoin ici, basées sur audit_keywords_definition
                    return 'Autre/Non Spécifié'

                # Appliquer la catégorisation une seule fois
                df_for_country_audit['audit_category'] = (df_for_country_audit.get('Lock reason', pd.Series(dtype=str)).fillna('') + " " + df_for_country_audit.get('Lock history', pd.Series(dtype=str)).fillna('')).str.lower().apply(get_audit_category)
                
                country_audit_counts = df_for_country_audit.groupby(['Country/Region', 'audit_category']).size().unstack(fill_value=0)
                # S'assurer que les colonnes existent même si aucun cas n'est trouvé pour un type
                if 'Integrity Program' not in country_audit_counts.columns: country_audit_counts['Integrity Program'] = 0
                if 'Autre/Non Spécifié' not in country_audit_counts.columns: country_audit_counts['Autre/Non Spécifié'] = 0
                country_audit_counts = country_audit_counts.reindex(top_10_countries_list).fillna(0) # Ordonner et remplir les NaN

                if not country_audit_counts.empty:
                    fig_stacked_bar = go.Figure()
                    colors_audit = {'Integrity Program': 'tomato', 'Autre/Non Spécifié': 'cornflowerblue'}
                    for audit_cat in ['Integrity Program', 'Autre/Non Spécifié']: # Ordre désiré pour l'empilement
                        if audit_cat in country_audit_counts.columns:
                            fig_stacked_bar.add_trace(go.Bar(y=country_audit_counts.index, x=country_audit_counts[audit_cat], name=audit_cat, orientation='h', marker_color=colors_audit.get(audit_cat)))
                    fig_stacked_bar.update_layout(barmode='stack', title_text="Types d'Audit par Pays (Top 10)", height=500, yaxis={'categoryorder':'total ascending'}, legend_title_text="Type d'Audit", title_x=0.5, paper_bgcolor='var(--card-background)', plot_bgcolor='var(--card-background)', font_color='var(--text-color)')
                    st.plotly_chart(fig_stacked_bar, use_container_width=True)
                
                st.markdown("---"); st.subheader("Tableau Détaillé des Suspensions par Pays")
                display_df_geo = geo_stats_df_filtered[['Country/Region', 'total_suspensions']]
                st.dataframe(display_df_geo.style.highlight_max(subset=['total_suspensions'], props='color:black; background-color:rgba(0,120,212,0.15); font-weight:bold;').format({'total_suspensions': '{:,}'}), use_container_width=True)
            else: st.info("Aucun pays avec des suspensions à afficher.")
        else: st.info("Données géographiques non disponibles.")

    with tab3:
        st.header("🏷️ Analyse Thématique Détaillée")
        if analyzer.locked_df is None or analyzer.locked_df.empty: st.warning("Aucune donnée à afficher."); return
        st.markdown("Explorez les motifs de suspension par thème.")
        theme_counts, theme_details = analyzer.analyze_themes()
        technical_themes = {k:v for k,v in theme_counts.items() if k not in ['ADMINISTRATIVE_OPERATIONAL_ISSUES', 'NON_CLASSIFIE']}
        admin_issues_count = theme_counts.get('ADMINISTRATIVE_OPERATIONAL_ISSUES', 0)
        unclassified_count = theme_counts.get('NON_CLASSIFIE', 0)

        st.subheader("Thèmes Techniques de Non-Conformité")
        if not technical_themes: st.info("Aucun thème technique identifié.")
        for theme, count in sorted(technical_themes.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                with st.expander(f"{theme.replace('_', ' ').title()} ({count} cas)", expanded=False):
                    st.markdown(f"**Exemples de motifs (jusqu'à 5) pour : {theme.replace('_', ' ').title()}**")
                    for i, detail in enumerate(theme_details.get(theme, [])[:5]):
                        st.markdown(f"**Cas {i+1} (Fournisseur: `{detail['supplier']}`, Pays: `{detail['country']}`)**")
                        reason_text = str(detail['reason'])
                        st.caption(f"{reason_text[:600]}...")
                        if len(reason_text) > 10:
                            translate_url = f"https://translate.google.com/?sl=auto&tl=fr&text={requests.utils.quote(reason_text[:1000])}"
                            st.markdown(f"<a href='{translate_url}' target='_blank' style='font-size:0.8em; color: var(--primary-color);'>Traduire ce motif...</a>", unsafe_allow_html=True)
                        if i < 4 : st.markdown("---")
        
        if admin_issues_count > 0:
            st.subheader("Problèmes Administratifs / Opérationnels")
            with st.expander(f"Problèmes Administratifs / Opérationnels ({admin_issues_count} cas)", expanded=True):
                for i, detail in enumerate(theme_details.get('ADMINISTRATIVE_OPERATIONAL_ISSUES', [])[:5]):
                    st.markdown(f"**Cas {i+1} (F: `{detail['supplier']}`, P: `{detail['country']}`)**"); st.caption(f"{str(detail['reason'])[:600]}...")
                    if i < 4 : st.markdown("---")
        if unclassified_count > 0:
            st.subheader("Motifs Non Classifiés")
            with st.expander(f"Non Classifiés ({unclassified_count} cas)", expanded=False):
                for i, detail in enumerate(theme_details.get('NON_CLASSIFIE', [])[:5]):
                    st.markdown(f"**Cas {i+1} (F: `{detail['supplier']}`, P: `{detail['country']}`)**"); st.caption(f"{str(detail['reason'])[:600]}...")
                    if i < 4 : st.markdown("---")

    with tab4:
        st.header("📋 Analyse des Exigences IFS Citées")
        if analyzer.locked_df is None or analyzer.locked_df.empty: st.warning("Aucune donnée à afficher."); return
        recommendations = analyzer.generate_ifs_recommendations_analysis()
        if recommendations and analyzer.checklist_df is not None:
            st.success("Checklist IFS Food V8 utilisée pour l'analyse des exigences.")
            df_reco = pd.DataFrame(recommendations).sort_values(by='count', ascending=False)
            top_reco_chart_df = df_reco.head(15).copy()
            if 'requirement_text' in top_reco_chart_df.columns: top_reco_chart_df['display_label'] = top_reco_chart_df.apply(lambda row: f"{row['chapter']} ({str(row['requirement_text'])[:30]}...)", axis=1)
            else: top_reco_chart_df['display_label'] = top_reco_chart_df['chapter']
            reco_chart_data = pd.Series(top_reco_chart_df['count'].values, index=top_reco_chart_df['display_label']).to_dict()
            if reco_chart_data: st.plotly_chart(analyzer._create_plotly_bar_chart(reco_chart_data, "Top 15 Exigences IFS Citées", orientation='v', color='gold', height=550, text_auto=False), use_container_width=True)
            st.markdown("---"); st.subheader("Détail des Exigences et Motifs Associés (Top 25)")
            for index, row_reco in df_reco.head(25).iterrows():
                with st.expander(f"Exigence {row_reco['chapter']} ({row_reco['count']} mentions)", expanded=False):
                    st.markdown(f"**Texte de l'exigence (Checklist IFS) :**\n\n> _{str(row_reco['requirement_text'])}_")
                    if row_reco.get('specific_reasons'):
                        st.markdown("---"); st.markdown("**Exemples de motifs de suspension liés à cette exigence (jusqu'à 3) :**")
                        for i, reason_detail in enumerate(row_reco['specific_reasons'][:3]):
                            st.caption(f"Cas {i+1} - F: `{reason_detail['supplier']}` (P: `{reason_detail['country']}`): {str(reason_detail['reason_text'])[:500]}...")
                            if i < 2: st.markdown("<br>", unsafe_allow_html=True)
        elif recommendations:
             st.warning("Checklist non chargée/valide. Affichage des numéros de chapitres uniquement.")
             df_reco_no_text = pd.DataFrame(recommendations).sort_values(by='count', ascending=False).head(15)
             chapter_counts_dict = pd.Series(df_reco_no_text['count'].values, index=df_reco_no_text['chapter']).to_dict()
             if chapter_counts_dict: st.plotly_chart(analyzer._create_plotly_bar_chart(chapter_counts_dict, "Top Chapitres IFS Cités (Numéros)", orientation='v', color='gold', height=500), use_container_width=True)
             st.dataframe(df_reco_no_text.rename(columns={'chapter':'Chapitre', 'count':'Mentions'}), use_container_width=True)
        else: st.info("Aucune exigence/chapitre IFS n'a pu être extrait, ou la checklist n'est pas disponible/utilisée.")

    with tab5:
        st.header("🕵️ Focus sur Audits Integrity Program")
        if analyzer.locked_df is None or analyzer.locked_df.empty: st.warning("Aucune donnée à afficher."); return
        audit_analysis, audit_examples = analyzer.analyze_audit_types()
        ip_theme_key = 'INTEGRITY_PROGRAM_IP'; ip_count = audit_analysis.get(ip_theme_key, 0)
        st.metric("Nombre total de cas liés à l'Integrity Program (IOC, On-site Check, etc.)", ip_count)

        if ip_count > 0 and ip_theme_key in audit_examples and audit_examples[ip_theme_key]['examples']:
            st.markdown(f"**Liste des {ip_count} motifs de suspension liés à l'Integrity Program :**")
            for i, ex_data in enumerate(audit_examples[ip_theme_key]['examples']):
                with st.expander(f"Cas IP {i+1} : {ex_data.get('Supplier', 'N/A')} ({ex_data.get('Country/Region', 'N/A')})", expanded=(i<1)): # Premier ouvert
                    st.caption(f"{str(ex_data.get('Lock reason', 'N/A'))}")
        elif ip_count > 0: st.info("Cas IP comptabilisés, mais pas d'exemples spécifiques trouvés (cela peut être dû à la limite de collecte d'exemples pour ce type d'audit).")
        else: st.info("Aucun cas 'Integrity Program' explicitement identifié.")
        
        st.markdown("---"); st.subheader("Répartition Générale des Autres Types d'Audit")
        other_audit_analysis = {k:v for k,v in audit_analysis.items() if k != ip_theme_key and v > 0}
        if other_audit_analysis:
            other_audit_analysis_clean = {k.replace('_', ' ').title():v for k,v in other_audit_analysis.items()}
            st.plotly_chart(analyzer._create_plotly_bar_chart(other_audit_analysis_clean, "Répartition des Autres Types d'Audit", color='darkorange', height=350), use_container_width=True)
        else: st.info("Aucun autre type d'audit spécifique identifié.")

    with tab6:
        st.header("🔗 Analyse Croisée : Thèmes Techniques vs Product Scopes")
        if analyzer.locked_df is None or analyzer.locked_df.empty: st.warning("Aucune donnée à afficher."); return
        cross_pivot_matrix = analyzer.cross_analysis_scope_themes()
        if cross_pivot_matrix is not None and not cross_pivot_matrix.empty:
            top_n_scopes_heatmap = min(15, len(cross_pivot_matrix.index))
            if len(cross_pivot_matrix.index) > top_n_scopes_heatmap:
                scope_totals = cross_pivot_matrix.sum(axis=1).sort_values(ascending=False)
                cross_pivot_matrix_filtered = cross_pivot_matrix.loc[scope_totals.head(top_n_scopes_heatmap).index]
            else: cross_pivot_matrix_filtered = cross_pivot_matrix

            if not cross_pivot_matrix_filtered.empty:
                 st.plotly_chart(analyzer._create_plotly_heatmap(cross_pivot_matrix_filtered, "Fréquence des Thèmes Techniques par Product Scope (Top Scopes)", height=max(500, len(cross_pivot_matrix_filtered.index) * 35 + 200)), use_container_width=True)
                 st.markdown("---"); st.subheader("Tableau de Corrélation Complet (Scopes vs Thèmes Techniques)")
                 st.dataframe(cross_pivot_matrix.style.background_gradient(cmap='Blues', axis=None).format("{:.0f}"), use_container_width=True)
            else: st.info("Pas assez de données pour la heatmap après filtrage.")
        else: st.info("Données insuffisantes pour l'analyse croisée Thèmes vs Product Scopes.")

# --- Exécution de l'application ---
if __name__ == "__main__":
    main()
