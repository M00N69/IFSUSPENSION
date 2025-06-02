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
import os # Pour la gestion de fichiers temporaires PDF
# import traceback # Pour un débogage plus poussé si nécessaire

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
        # Définition unique des thèmes et de leurs mots-clés, incluant des numéros de chapitres pertinents
        self.themes_keywords_definition = {
            'HYGIENE_PERSONNEL': ['hygien', 'personnel', 'clothing', 'hand', 'wash', 'uniform', 'gmp', 'locker', 'changing room', 'work clothing', 'protective clothes', '3.2.1', '3.2.2', '3.2.7', '3.2.9', '3.2.10', '3.4.2'],
            'HACCP_CCP_OPRP': ['haccp', 'ccp', 'oprp', 'critical', 'control', 'point', 'monitoring', 'hazard analysis', 'validation haccp', '2.3.9.1', '2.3.9.2', '2.3.9.4', '2.3.11.1', '5.3.2'],
            'TRACEABILITY': ['traceability', 'trace', 'record', 'batch', 'lot', 'identification', 'tracking', '4.18.1'],
            'ALLERGEN_MANAGEMENT': ['allergen', 'allergy', 'cross-contamination', 'gluten', 'lactose', 'celery', 'mustard', 'wheat', 'egg', '4.19.2'],
            'PEST_CONTROL': ['pest', 'rodent', 'insect', 'trap', 'bait', 'infestation', 'fly', 'mouse', 'rat', 'moth', 'weevil', 'spider', 'cobweb', '4.13.1', '4.13.2', '4.13.4', '4.13.5', '4.13.7'],
            'CLEANING_SANITATION': ['clean', 'sanitation', 'disinfect', 'chemical', 'cleaning plan', 'dirt', 'residue', '4.10.1', '4.10.2', '4.10.6'],
            'TEMPERATURE_CONTROL': ['temperature', 'cold', 'heat', 'refrigerat', 'freez', 'thaw', 'cooling'], # Numéros de chapitres spécifiques peuvent être ajoutés si identifiés
            'MAINTENANCE_EQUIPMENT': ['maintenance', 'equipment', 'calibrat', 'repair', 'infrastructure', 'facility', 'structure', 'conveyor', '4.9.1.1', '4.9.2.2', '4.9.2.3', '4.9.3.1', '4.9.4.1', '4.9.6.2','4.16.5', '4.17.2', '4.17.4'],
            'DOCUMENTATION_RECORDS': ['document', 'procedure', 'record', 'manual', 'specification', 'not documented', '5.1.1', '5.1.2', '1.3.1', '2.1.2.1', '2.2.3.8', '2.2.3.9', '2.3.5.1'],
            'FOREIGN_BODY_CONTAMINATION': ['foreign body', 'foreign material', 'glass', 'metal', 'detect', 'x-ray', 'contaminat', 'wood', 'plastic', 'paper', 'paint', 'ink', '4.12.1', '4.12.2', '4.12.3'],
            'STORAGE_WAREHOUSING': ['storage', 'warehouse', 'stock', 'segregat', 'pallet', 'raw material storage', '4.14.3', '4.14.5', '4.15.1', '4.15.6'],
            'SUPPLIER_RAW_MATERIAL_CONTROL': ['supplier', 'vendor', 'purchase', 'raw material', 'ingredient', 'packaging material', 'declaration of conformity', 'doc', '4.5.1', '4.5.2', '4.2.1.2', '4.2.1.3', '4.6.2', '4.6.3', '4.7.1.2'],
            'LABELLING': ['label', 'labelling', 'declaration', 'ingredient list', 'mrl', 'allergen labelling', 'nutritional information', '4.3.1', '4.3.2', '4.3.5', '4.2.1.1'],
            'QUANTITY_CONTROL_WEIGHT': ['quantity control', 'weight', 'fill', 'scale', 'metrological', 'underfilling', '5.5.1', '5.5.2', '5.4.1', '5.4.2', '5.10.3'],
            'MANAGEMENT_RESPONSIBILITY_CULTURE': ['management', 'responsibilit', 'food safety culture', 'internal audit', 'corrective action', 'training', '1.1.1', '1.1.2', '1.2.1', '1.2.4', '2.2.1.1', '2.3.6.1','3.1.2', '4.8.1', '4.8.2', '4.8.4', '4.8.5', '4.9.4','5.2.1', '5.6.1', '5.6.2', '5.7.2','5.11.1', '5.11.2', '5.11.3', '5.11.4'],
            'NON_PAYMENT_ADMINISTRATIVE': ['payment', 'invoice', 'pay', 'closure', 'discontinued', 'bankrupt', 'denies access', 'ceased operation', 'fire', 'merged', 'cessation of activity', 'discontinuance of business'],
            'INTEGRITY_PROGRAM_ISSUES': ['integrity', 'on-site check', 'ioc', 'unannounced audit', 'integrity on-site check', 'integrity on site check']
        }
        self.load_data(locked_file_io, checklist_file_io)
        if self.locked_df is not None:
            self.clean_lock_reasons()

    def load_data(self, locked_file_io, checklist_file_io=None):
        try:
            self.locked_df = pd.read_csv(locked_file_io, encoding='utf-8')
            if 'Standard' in self.locked_df.columns:
                self.locked_df = self.locked_df[self.locked_df['Standard'].str.contains('IFS Food', na=False, case=False)]
            if checklist_file_io:
                try:
                    self.checklist_df = pd.read_csv(checklist_file_io, encoding='utf-8')
                    if 'Requirement Number' not in self.checklist_df.columns or \
                       'Requirement text (English)' not in self.checklist_df.columns:
                        st.warning("Colonnes 'Requirement Number' ou 'Requirement text (English)' manquantes dans la checklist. L'analyse des exigences sera limitée.")
                        self.checklist_df = None
                except Exception as e_checklist:
                    st.error(f"Erreur lors du chargement du fichier checklist : {e_checklist}")
                    self.checklist_df = None
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement du fichier des suspensions : {e}")
            self.locked_df = None

    def clean_lock_reasons(self):
        if self.locked_df is None or 'Lock reason' not in self.locked_df.columns: return
        self.locked_df['lock_reason_clean'] = self.locked_df['Lock reason'].astype(str).fillna('') \
            .str.lower() \
            .str.replace(r'[\n\r\t]', ' ', regex=True) \
            .str.replace(r'[^\w\s\.\-\/\%]', ' ', regex=True) \
            .str.replace(r'\s+', ' ', regex=True).str.strip()

    def extract_ifs_chapters(self, text):
        if pd.isna(text) or not isinstance(text, str) or text.strip() == '': return []
        patterns = [
            r'(?:ko|major|cl\.|req\.|requirement(?: item)?|chapter|section|point|§|cl\s+|clause)?\s*(\d\.\d{1,2}(?:\.\d{1,2})?)',
            r'(\d\.\d{1,2}(?:\.\d{1,2})?)\s*(?:ko|major|:|-|\(ko\)|\(major\))',
            r'(\d{1,2})\s*-\s*ko', # Ex: 5.11.3 - KO (capture le numéro seul)
            r'requirement\s+(\d\.\d\.\d)',
            r'cl\s+(\d\.\d+(?:\.\d+)?)', # Ex: cl 4.12.1
            r'§\s*(\d\.\d+(?:\.\d+)?)'   # Ex: § 4.13.1
        ]
        chapters_found = []
        normalized_text = text.lower().replace('\n', ' ').replace('\r', ' ')
        for pattern in patterns:
            matches = re.findall(pattern, normalized_text)
            for match in matches:
                chapter_num_match = match if isinstance(match, str) else (match[-1] if isinstance(match, tuple) and match[-1] else match[0] if isinstance(match, tuple) and match[0] else None)
                if chapter_num_match:
                    chapter_num = str(chapter_num_match).strip().rstrip('.').strip()
                    if re.fullmatch(r'\d(\.\d+){1,2}', chapter_num) or re.fullmatch(r'\d\.\d+', chapter_num) :
                        main_chapter_part = chapter_num.split('.')[0]
                        if main_chapter_part.isdigit() and 1 <= int(main_chapter_part) <= 6: # IFS Food v8 chapters 1-6
                             chapters_found.append(chapter_num)
        return sorted(list(set(chapters_found)))

    def analyze_themes(self):
        if self.locked_df is None or 'lock_reason_clean' not in self.locked_df.columns: return {}, {}
        theme_counts = {theme: 0 for theme in self.themes_keywords_definition}
        theme_details = {theme: [] for theme in self.themes_keywords_definition}
        for index, row in self.locked_df.iterrows():
            reason_text = row['lock_reason_clean']
            original_reason = row['Lock reason']
            supplier = row.get('Supplier', 'N/A')
            country = row.get('Country/Region', 'N/A')
            for theme, keywords in self.themes_keywords_definition.items():
                # Combinaison des mots-clés textuels et des numéros de chapitres extraits des keywords
                # (les numéros de chapitre sont déjà dans les keywords de self.themes_keywords_definition)
                if any(keyword in reason_text for keyword in keywords):
                    theme_counts[theme] += 1
                    theme_details[theme].append({
                        "reason": original_reason,
                        "supplier": supplier,
                        "country": country
                    })
        return theme_counts, theme_details

    def geographic_analysis(self):
        if self.locked_df is None or 'Country/Region' not in self.locked_df.columns: return None
        return self.locked_df.groupby('Country/Region').size().sort_values(ascending=False).reset_index(name='total_suspensions')

    def clean_product_scopes(self, scope_text):
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

    def product_scope_analysis(self):
        if self.locked_df is None or 'Product scopes' not in self.locked_df.columns: return None
        all_scopes = []
        for scopes_text in self.locked_df['Product scopes'].dropna():
            all_scopes.extend(self.clean_product_scopes(scopes_text))
        return Counter(all_scopes)

    def chapter_frequency_analysis(self):
        if self.locked_df is None or 'Lock reason' not in self.locked_df.columns: return Counter()
        all_chapters = []
        for reason in self.locked_df['Lock reason'].dropna():
            all_chapters.extend(self.extract_ifs_chapters(reason))
        return Counter(all_chapters)

    def analyze_audit_types(self):
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
        chapter_counts = self.chapter_frequency_analysis()
        if not chapter_counts: return None
        recommendations = []
        for chapter, count in chapter_counts.most_common():
            norm_chapter = chapter.replace("KO ", "").strip()
            req_text_series = self.checklist_df[self.checklist_df['Requirement Number'].astype(str).str.strip() == norm_chapter]['Requirement text (English)']
            req_text = req_text_series.iloc[0] if not req_text_series.empty else "Texte de l'exigence non trouvé dans la checklist fournie."
            recommendations.append({'chapter': chapter, 'count': count, 'requirement_text': req_text})
        return recommendations

    def cross_analysis_scope_themes(self):
        if self.locked_df is None or 'Product scopes' not in self.locked_df.columns or 'lock_reason_clean' not in self.locked_df.columns: return None
        # Utiliser les thèmes principaux pour la lisibilité de la heatmap
        themes_for_cross = {
            'HYGIENE': self.themes_keywords_definition['HYGIENE_PERSONNEL'], 'HACCP': self.themes_keywords_definition['HACCP_CCP_OPRP'],
            'TRACE': self.themes_keywords_definition['TRACEABILITY'], 'ALLERGEN': self.themes_keywords_definition['ALLERGEN_MANAGEMENT'],
            'CLEAN': self.themes_keywords_definition['CLEANING_SANITATION'], 'MAINT': self.themes_keywords_definition['MAINTENANCE_EQUIPMENT'],
            'LABEL': self.themes_keywords_definition['LABELLING'], 'PEST': self.themes_keywords_definition['PEST_CONTROL'],
            'MGT_SYS': self.themes_keywords_definition['MANAGEMENT_RESPONSIBILITY_CULTURE'], # Management System
            'F_BODY': self.themes_keywords_definition['FOREIGN_BODY_CONTAMINATION'] # Foreign Body
        }
        scope_theme_data = []
        for idx, row in self.locked_df.iterrows():
            scopes_text, reason_text = row['Product scopes'], row['lock_reason_clean']
            if pd.notna(scopes_text) and pd.notna(reason_text) and reason_text:
                for scope in self.clean_product_scopes(scopes_text):
                    for theme, keywords in themes_for_cross.items():
                        if any(kw in reason_text for kw in keywords):
                            scope_theme_data.append({'scope': f"Scope {scope}", 'theme': theme})
        if not scope_theme_data: return None
        return pd.DataFrame(scope_theme_data).pivot_table(index='scope', columns='theme', aggfunc='size', fill_value=0)

    def _create_plotly_bar_chart(self, data_dict, title, orientation='v', xaxis_title="", yaxis_title="", color='royalblue', height=400, text_auto=True):
        if not data_dict : return go.Figure()
        y_data, x_data = (list(data_dict.keys()), list(data_dict.values())) if orientation == 'h' else (list(data_dict.values()), list(data_dict.keys()))
        fig = go.Figure(go.Bar(x=x_data, y=y_data, orientation=orientation, marker_color=color, text=y_data if orientation=='v' else x_data, textposition='auto' if text_auto else None))
        fig.update_layout(title={'text': f"<b>{title}</b>", 'x':0.5, 'font': {'size': 16, 'family': "Arial, sans-serif"}},
                          xaxis_title=xaxis_title, yaxis_title=yaxis_title, height=height,
                          margin=dict(l=20, r=20, t=60, b=20), font=dict(family="Arial, sans-serif", size=10),
                          yaxis=dict(autorange="reversed", tickfont_size=9) if orientation == 'h' else dict(tickfont_size=9),
                          xaxis=dict(tickfont_size=9),
                          plot_bgcolor='rgba(245,245,245,1)', # Correction de 245f à 245
                          paper_bgcolor='rgba(255,255,255,1)')
        return fig

    def _create_plotly_choropleth_map(self, geo_data_df, title, height=500):
        if geo_data_df is None or geo_data_df.empty: return go.Figure()
        fig = px.choropleth(geo_data_df, locations="Country/Region", locationmode='country names',
                            color="total_suspensions", hover_name="Country/Region",
                            color_continuous_scale=px.colors.sequential.Viridis_r,
                            title=title, height=height)
        fig.update_layout(title={'text': f"<b>{title}</b>", 'x':0.5, 'font': {'size': 16, 'family': "Arial, sans-serif"}},
                          geo=dict(showframe=False, showcoastlines=True, projection_type='natural earth', bgcolor='rgba(230,240,255,1)'), # Pas de 'f' ici
                          margin=dict(l=10, r=10, t=50, b=10), font=dict(family="Arial, sans-serif"),
                          paper_bgcolor='rgba(255,255,255,1)')
        return fig

    def _create_plotly_heatmap(self, pivot_matrix, title, height=500):
        if pivot_matrix is None or pivot_matrix.empty: return go.Figure()
        fig = px.imshow(pivot_matrix, text_auto='.0f', aspect="auto", color_continuous_scale='Blues', title=title, height=height)
        fig.update_layout(title={'text': f"<b>{title}</b>", 'x':0.5, 'font': {'size': 16, 'family': "Arial, sans-serif"}},
                          margin=dict(l=10, r=10, t=80, b=10), font=dict(family="Arial, sans-serif"),
                          xaxis=dict(tickangle=30, side='bottom', tickfont_size=9), yaxis=dict(tickfont_size=9),
                          paper_bgcolor='rgba(255,255,255,1)')
        fig.update_traces(hovertemplate="Scope: %{y}<br>Thème: %{x}<br>Cas: %{z}<extra></extra>")
        return fig

    def _add_text_to_pdf_page(self, fig, text_lines, start_y=0.95, line_height=0.035, font_size=9, title="", title_font_size=14, max_chars_per_line=100):
        ax = fig.gca(); ax.clear(); ax.axis('off')
        if title:
            ax.text(0.5, start_y, title, ha='center', va='top', fontsize=title_font_size, fontweight='bold', fontname='DejaVu Sans')
            start_y -= (line_height * 2.5)
        current_y = start_y
        for line in text_lines:
            import textwrap
            wrapped_lines = textwrap.wrap(line, width=max_chars_per_line, break_long_words=False, replace_whitespace=False) # break_long_words=False
            for wrapped_line in wrapped_lines:
                if current_y < 0.05: return False
                fw = 'bold' if line.startswith(tuple(["🎯","📊","🌍","🏭","📋","🔍", "---"])) else 'normal'
                fs = font_size + 1 if fw == 'bold' else font_size
                if line.startswith("---"): fs -=1
                ax.text(0.03, current_y, wrapped_line, ha='left', va='top', fontsize=fs, fontweight=fw, fontname='DejaVu Sans')
                current_y -= line_height
            if not line.strip(): current_y -= (line_height * 0.3)
        return True

    def _create_matplotlib_figure_for_pdf(self, data_dict_or_df, title, x_label="", y_label="", chart_type='barh', top_n=10, color='skyblue', xtick_rotation=0, ytick_fontsize=8):
        if not data_dict_or_df and not isinstance(data_dict_or_df, pd.DataFrame) : return None
        fig, ax = plt.subplots(figsize=(10, 6.5))
        items, values = [], []

        if isinstance(data_dict_or_df, (Counter, dict)):
            # Filtrer les éléments avec une valeur nulle avant de trier
            filtered_data = {k: v for k, v in data_dict_or_df.items() if v > 0}
            if not filtered_data: return None # Si tout est nul après filtrage
            sorted_data = dict(sorted(filtered_data.items(), key=lambda item: item[1], reverse=True)[:top_n])
            items = [str(k).replace('_',' ').replace('MANAGEMENT','MGMT').replace('RESPONSIBILITY','RESP.')[:35] for k in sorted_data.keys()]
            values = list(sorted_data.values())
        elif isinstance(data_dict_or_df, pd.DataFrame):
            # S'assurer que les colonnes existent avant de les utiliser
            df_top = data_dict_or_df.head(top_n)
            if 'Country/Region' in df_top.columns and 'total_suspensions' in df_top.columns:
                items = df_top['Country/Region'].tolist(); values = df_top['total_suspensions'].tolist(); chart_type = 'bar'
            elif 'chapter' in df_top.columns and 'count' in df_top.columns and 'requirement_text' in df_top.columns:
                 # Filtrer pour les valeurs de 'count' > 0
                 df_top_filtered = df_top[df_top['count'] > 0]
                 if df_top_filtered.empty: return None
                 items = [f"{row['chapter']}\n({row['requirement_text'][:40]}...)" if row['requirement_text'] != "Texte de l'exigence non trouvé dans la checklist fournie." else row['chapter'] for index, row in df_top_filtered.iterrows()]
                 values = df_top_filtered['count'].tolist(); chart_type = 'bar'
            else:
                if not df_top.empty:
                    # S'assurer qu'il y a au moins deux colonnes pour ce cas générique, ou une pour index/valeur
                    if len(df_top.columns) >= 1:
                         items = df_top.index.astype(str).tolist() if len(df_top.columns) == 1 else df_top.iloc[:,0].astype(str).tolist()
                         values = df_top.iloc[:,0].tolist() if len(df_top.columns) == 1 else df_top.iloc[:,1].tolist()
                         values = [v for v in values if v > 0] # Filtrer valeurs nulles
                         if not values: return None
                         items = items[:len(values)] # S'assurer que items et values ont la même longueur
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

    def export_report_to_pdf(self, filename='IFS_Analysis_Report.pdf'):
        if self.locked_df is None: return None
        try:
            with PdfPages(filename) as pdf:
                total_suspensions = len(self.locked_df)
                if total_suspensions == 0:
                    fig = plt.figure(figsize=(8.5, 11)); self._add_text_to_pdf_page(fig, ["Aucune donnée à analyser."], title="Rapport d'Analyse IFS"); pdf.savefig(fig); plt.close(fig); return filename

                # Page 1: Couverture
                fig = plt.figure(figsize=(8.5, 11))
                ln_o = st.session_state.get('locked_file_name_original', 'N/A')
                cn_o = st.session_state.get('checklist_file_name_original', 'Non fournie')
                title_text = [f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}", "", f"Fichier Suspensions: {ln_o}", f"Fichier Checklist: {cn_o}", "", "📊 VUE D'ENSEMBLE"]
                title_text.append(f"   • Total suspensions IFS Food: {total_suspensions}")
                wr_c = self.locked_df['Lock reason'].notna().sum(); title_text.append(f"   • Avec motifs: {wr_c} ({wr_c/total_suspensions*100:.1f}% si total > 0 else 0%)")
                audit_s_sum, _ = self.analyze_audit_types(); total_as = sum(audit_s_sum.values()); title_text.append(f"   • Liées à audits spécifiques: {total_as} ({total_as/total_suspensions*100:.1f}% si total > 0 else 0%)")
                self._add_text_to_pdf_page(fig, title_text, title="Rapport d'Analyse IFS Food Safety"); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

                # Graphiques
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
                    for r_ in sorted(reco, key=lambda x: x['count'], reverse=True)[:25]: # Afficher plus dans le PDF
                        req_tl.extend([f"📋 Chap {r_['chapter']} ({r_['count']} mentions)", f"   Txt: {r_['requirement_text']}", ""])
                    self._add_text_to_pdf_page(fig, req_tl, title="Détail Exigences IFS", line_height=0.025, font_size=6, max_chars_per_line=130)
                    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
            return filename
        except Exception as e:
            st.error(f"❌ Erreur majeure lors de la génération du PDF: {e}")
            # traceback.print_exc()
            return None

    def generate_detailed_theme_analysis_text(self):
        if self.locked_df is None: return ""
        theme_counts, theme_details = self.analyze_themes()
        lines = []
        for theme, count in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                lines.append(f"\n🎯 {theme.replace('_', ' ').title()} ({count} cas)")
                lines.append("-" * 60)
                for i, detail in enumerate(theme_details[theme][:3]): # 3 exemples pour le PDF
                    reason_short = detail['reason'][:200] + "..." if len(detail['reason']) > 200 else detail['reason']
                    lines.append(f"   Ex {i+1} ({detail['supplier']}, {detail['country']}):")
                    lines.append(f"     Motif: {reason_short}")
                lines.append("") # Espace
        return "\n".join(lines)

    def generate_audit_analysis_report_text(self):
        if self.locked_df is None: return ""
        audit_analysis, audit_examples = self.analyze_audit_types()
        total_suspensions = len(self.locked_df)
        if total_suspensions == 0: return "Aucune suspension à analyser."
        lines = [f"Total audits spécifiques: {sum(audit_analysis.values())} ({sum(audit_analysis.values())/total_suspensions*100:.1f}% du total des suspensions, si total > 0 else 0%)"]
        for audit_type, count in sorted(audit_analysis.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                lines.append(f"\n🎯 {audit_type.replace('_', ' ').title()} ({count} cas - {count/total_suspensions*100:.1f}%)")
                lines.append("-" * 60)
                for i, ex_data in enumerate(audit_examples[audit_type]['examples'][:2]): # 2 exemples pour le PDF
                    reason_short = ex_data.get('Lock reason', 'N/A')[:200] + "..." if len(ex_data.get('Lock reason', 'N/A')) > 200 else ex_data.get('Lock reason', 'N/A')
                    lines.append(f"   Ex {i+1} ({ex_data.get('Supplier', 'N/A')}, {ex_data.get('Country/Region', 'N/A')}):")
                    lines.append(f"     Motif: {reason_short}")
                lines.append("") # Espace
        return "\n".join(lines)

# --- Fonctions Utilitaires pour Streamlit ---
@st.cache_resource
def get_analyzer_instance(_locked_data_io, _checklist_data_io, locked_file_key, checklist_file_key):
    return IFSAnalyzer(_locked_data_io, _checklist_data_io)

@st.cache_data(ttl=3600) # Cache pour 1 heure
def download_checklist_content_from_github(url): # Renommée pour clarté
    try:
        response = requests.get(url, timeout=15) # Augmenter timeout
        response.raise_for_status()
        return response.text # Retourner le texte brut
    except requests.exceptions.RequestException as e:
        st.warning(f"Impossible de télécharger la checklist depuis GitHub ({e}). L'analyse des exigences sera limitée.")
        return None

# --- Interface Streamlit ---
def main():
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
            # L'URL est fixée ici
            checklist_url = "https://raw.githubusercontent.com/M00N69/Action-plan/main/Guide%20Checklist_IFS%20Food%20V%208%20-%20CHECKLIST.csv"
            checklist_text_content = download_checklist_content_from_github(checklist_url)
            if checklist_text_content:
                checklist_data_io = io.StringIO(checklist_text_content) # Utiliser StringIO pour du texte
                current_checklist_file_key = f"github_checklist_content_hash_{hash(checklist_text_content)}" # Clé basée sur le contenu
            else: # Échec du téléchargement
                 current_checklist_file_key = "github_checklist_failed"
            st.session_state.checklist_file_name_original = "Checklist IFS Food V8 (GitHub)"
        else:
            st.session_state.checklist_file_name_original = "Non fournie"

        analyzer = get_analyzer_instance(locked_data_io, checklist_data_io, current_locked_file_key, current_checklist_file_key)

        if analyzer.locked_df is not None and not analyzer.locked_df.empty:
            st.success(f"Fichier **'{locked_file_uploaded.name}'** analysé : **{len(analyzer.locked_df)}** suspensions IFS Food trouvées.")
            display_dashboard_tabs(analyzer)

            st.sidebar.markdown("---")
            st.sidebar.subheader("Téléchargement du Rapport")
            if st.sidebar.button("📄 Générer et Télécharger le PDF", key="pdf_button_main", help="Génère un rapport PDF complet avec graphiques et analyses détaillées."):
                with st.spinner("Génération du rapport PDF en cours... Cela peut prendre quelques instants."):
                    temp_pdf_filename = f"temp_report_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.pdf" # Nom de fichier plus unique
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
                        except Exception: pass # Ne pas bloquer si la suppression échoue
                    else:
                        st.sidebar.error("Erreur lors de la création du rapport PDF. Veuillez vérifier les messages d'erreur dans la console si vous exécutez localement, ou les logs sur Streamlit Cloud.")
        else:
            st.error("Aucune donnée IFS Food n'a été trouvée dans le fichier téléversé ou après filtrage. Veuillez vérifier son contenu et son format.")
    else:
        st.info("👈 Veuillez téléverser un fichier CSV des suspensions IFS via la barre latérale pour commencer l'analyse.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("Application d'analyse IFS")
    st.sidebar.markdown("Version 1.1")


def display_dashboard_tabs(analyzer):
    tab_titles = [
        "📊 Vue d'Ensemble", "🌍 Géographie", "🏷️ Thèmes Détaillés",
        "📋 Exigences IFS", "🕵️ Audits Spécifiques", "🔗 Analyse Croisée"
    ]
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_titles)

    with tab1:
        st.header("📊 Vue d'Ensemble des Suspensions")
        col1, col2, col3 = st.columns(3)
        total_suspensions = len(analyzer.locked_df)
        with_reasons_count = analyzer.locked_df['Lock reason'].notna().sum()
        audit_analysis_summary, _ = analyzer.analyze_audit_types()
        total_audit_special = sum(audit_analysis_summary.values())

        with col1: st.metric("Total Suspensions IFS Food", total_suspensions, help="Nombre total de suspensions après filtrage pour 'IFS Food'.")
        with col2: st.metric("Avec Motifs Documentés", f"{with_reasons_count} ({with_reasons_count/total_suspensions*100:.1f}% si total > 0 else 0%)", help="Pourcentage de suspensions ayant un motif renseigné.")
        with col3: st.metric("Liées à Audits Spécifiques", f"{total_audit_special} ({total_audit_special/total_suspensions*100:.1f}% si total > 0 else 0%)", help="Suspensions liées à des audits (Integrity Program, surveillance, etc.).")

        st.markdown("---")
        st.subheader("Visualisations Clés")
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

    with tab2:
        st.header("🌍 Analyse Géographique")
        geo_stats_df = analyzer.geographic_analysis()
        if geo_stats_df is not None and not geo_stats_df.empty:
            st.plotly_chart(analyzer._create_plotly_choropleth_map(geo_stats_df, "Suspensions par Pays"), use_container_width=True)
            st.markdown("---")
            st.subheader("Tableau des Suspensions par Pays (Top 20)")
            st.dataframe(geo_stats_df.head(20).style.highlight_max(subset=['total_suspensions'], color='rgba(255,170,170,0.5)', axis=0)
                                                    .format({'total_suspensions': '{:,}'}), use_container_width=True)
        else: st.info("Données géographiques non disponibles ou insuffisantes.")

    with tab3:
        st.header("🏷️ Analyse Thématique Détaillée")
        st.markdown("Explorez les motifs de suspension par thème. Cliquez sur un thème pour voir des exemples.")
        theme_counts, theme_details = analyzer.analyze_themes()
        for theme, count in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                with st.expander(f"{theme.replace('_', ' ').title()} ({count} cas)", expanded=False):
                    st.markdown(f"**Exemples de motifs (jusqu'à 5) pour le thème : {theme.replace('_', ' ').title()}**")
                    for i, detail in enumerate(theme_details[theme][:5]):
                        st.markdown(f"**Cas {i+1} (Fournisseur: `{detail['supplier']}`, Pays: `{detail['country']}`)**")
                        st.caption(f"{detail['reason'][:600]}...")
                        if i < 4 : st.markdown("---")
                    theme_keywords_current_theme = analyzer.themes_keywords_definition.get(theme, [])
                    if theme_keywords_current_theme: # S'assurer que le thème existe dans la définition
                        theme_mask = analyzer.locked_df['lock_reason_clean'].str.contains('|'.join(theme_keywords_current_theme), case=False, na=False, regex=True)
                        if theme_mask.sum() > 0:
                            theme_countries_df = analyzer.locked_df[theme_mask]['Country/Region'].value_counts().reset_index().head(5)
                            theme_countries_df.columns = ['Pays', 'Nb Cas']
                            if not theme_countries_df.empty:
                                st.markdown("**Pays les plus affectés par ce thème :**")
                                st.table(theme_countries_df)

    with tab4:
        st.header("📋 Analyse des Exigences IFS")
        recommendations = analyzer.generate_ifs_recommendations_analysis()
        if recommendations and analyzer.checklist_df is not None:
            st.success("Checklist IFS Food V8 utilisée pour l'analyse des exigences.")
            df_reco = pd.DataFrame(recommendations).sort_values(by='count', ascending=False)
            top_reco_chart_df = df_reco.head(15).copy()
            top_reco_chart_df['display_label'] = top_reco_chart_df.apply(lambda row: f"{row['chapter']} ({row['requirement_text'][:30]}...)", axis=1)
            reco_chart_data = pd.Series(top_reco_chart_df['count'].values, index=top_reco_chart_df['display_label']).to_dict()
            st.plotly_chart(analyzer._create_plotly_bar_chart(reco_chart_data, "Top 15 Exigences IFS Citées", orientation='v', color='gold', height=550, text_auto=False), use_container_width=True)
            st.markdown("---")
            st.subheader("Détail des Exigences Citées (Top 25)")
            for index, row in df_reco.head(25).iterrows():
                with st.expander(f"Exigence {row['chapter']} ({row['count']} mentions)", expanded=False):
                    st.markdown(f"**Texte de l'exigence :**\n\n> _{row['requirement_text']}_")
        elif recommendations:
             st.warning("Checklist non chargée. Affichage des numéros de chapitres uniquement.")
             df_reco_no_text = pd.DataFrame(recommendations).sort_values(by='count', ascending=False).head(15)
             chapter_counts_dict = pd.Series(df_reco_no_text['count'].values, index=df_reco_no_text['chapter']).to_dict()
             st.plotly_chart(analyzer._create_plotly_bar_chart(chapter_counts_dict, "Top Chapitres IFS Cités (Numéros)", orientation='v', color='gold', height=500), use_container_width=True)
             st.dataframe(df_reco_no_text, use_container_width=True)
        else:
            st.info("Aucune exigence/chapitre IFS spécifique n'a pu être extrait, ou la checklist n'est pas disponible/utilisée.")

    with tab5:
        st.header("🕵️ Analyse par Audits Spécifiques")
        audit_analysis, audit_examples = analyzer.analyze_audit_types()
        if audit_analysis:
            audit_analysis_clean = {k.replace('_', ' ').title():v for k,v in audit_analysis.items() if v > 0}
            if audit_analysis_clean: st.plotly_chart(analyzer._create_plotly_bar_chart(audit_analysis_clean, "Répartition par Type d'Audit Spécifique", color='darkorange', height=400), use_container_width=True)
            st.markdown("---")
            st.subheader("Détails et Exemples par Type d'Audit")
            for audit_type, count in sorted(audit_analysis.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    with st.expander(f"{audit_type.replace('_', ' ').title()} ({count} cas)", expanded=False):
                        st.markdown(f"**Exemples (jusqu'à 5) pour : {audit_type.replace('_', ' ').title()}**")
                        for i, ex_data in enumerate(audit_examples[audit_type]['examples'][:5]):
                            st.markdown(f"**Cas {i+1} (Fournisseur: `{ex_data.get('Supplier', 'N/A')}`, Pays: `{ex_data.get('Country/Region', 'N/A')}`)**")
                            st.caption(f"{ex_data.get('Lock reason', 'N/A')[:600]}...")
                            if i < 4 : st.markdown("---")
                        countries_data = audit_examples[audit_type]['countries']
                        if countries_data:
                            st.markdown(f"**Répartition géographique (Top 5 pays) :** {', '.join([f'{c} ({n})' for c, n in countries_data.items()])}")
        else: st.info("Aucune donnée sur les types d'audits spécifiques disponible.")

    with tab6:
        st.header("🔗 Analyse Croisée : Thèmes vs Product Scopes")
        cross_pivot_matrix = analyzer.cross_analysis_scope_themes()
        if cross_pivot_matrix is not None and not cross_pivot_matrix.empty:
            top_n_scopes_heatmap = min(15, len(cross_pivot_matrix.index))
            if len(cross_pivot_matrix.index) > top_n_scopes_heatmap:
                scope_totals = cross_pivot_matrix.sum(axis=1).sort_values(ascending=False)
                cross_pivot_matrix_filtered = cross_pivot_matrix.loc[scope_totals.head(top_n_scopes_heatmap).index]
            else: cross_pivot_matrix_filtered = cross_pivot_matrix

            if not cross_pivot_matrix_filtered.empty:
                 st.plotly_chart(analyzer._create_plotly_heatmap(cross_pivot_matrix_filtered, "Fréquence des Thèmes par Product Scope (Top Scopes)", height=max(500, len(cross_pivot_matrix_filtered.index) * 35 + 200)), use_container_width=True) # Augmenté la hauteur de base
                 st.markdown("---")
                 st.subheader("Tableau de Corrélation Complet (Scopes vs Thèmes)")
                 st.dataframe(cross_pivot_matrix.style.background_gradient(cmap='Blues', axis=None).format("{:.0f}"), use_container_width=True)
            else: st.info("Pas assez de données pour la heatmap après filtrage.")
        else: st.info("Données insuffisantes pour l'analyse croisée Thèmes vs Product Scopes.")

# --- Exécution de l'application ---
if __name__ == "__main__":
    main()
