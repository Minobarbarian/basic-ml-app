"""
Aplicação principal do Streamlit (View).
Este arquivo lida apenas com a interface do usuário.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from view import api_client
from view.api_client import APIConnectionError, APIError


def render_prediction(model_name: str, details: dict, full_data: dict):
    """
    Função auxiliar para renderizar o resultado de um único modelo na UI.
    """
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric(
            label=f"Modelo: {model_name}", 
            value=f"Intenção: {details['top_intent']}"
        )
        st.info(f"Texto original: `{full_data.get('text')}`")
        st.caption(f"Dono (Owner): `{full_data.get('owner')}`")

    with col2:
        st.markdown(f"**Probabilidades (Modelo: {model_name})**")
        
        probs_dict = details['all_probs']

        df_probs = pd.DataFrame.from_dict(
            probs_dict, 
            orient='index', 
            columns=['Probabilidade']
        ).reset_index()

        df_probs = df_probs.rename(columns={'index': 'Intenção'})
        df_probs = df_probs.sort_values(by='Probabilidade', ascending=False)
        
        fig = px.pie(
            df_probs, 
            values='Probabilidade', 
            names='Intenção',
            title=f"Probabilidades: {model_name}",
            hole=0.3,
        )

        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=False, margin=dict(t=30, b=0, l=0, r=0))

        st.plotly_chart(fig, width='stretch')

# --- Configuração da Página (View) ---
st.set_page_config(
    page_title="Cliente da API de Classificação",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Cliente para API de Classificação de Intenção")
st.markdown("Insira um texto abaixo para classificá-lo usando a API MLOps.")

# --- Interface do Usuário (View) ---
with st.form("predict_form"):
    text_input = st.text_area("Texto para classificar:", "please help me")
    submit_button = st.form_submit_button(label="Classificar Texto")

# --- Lógica de "Controle" (Controller) ---
if submit_button and text_input:
    # 1. Mostra um spinner
    with st.spinner("Classificando... 🧠"):
        try:
            # 2. Chama o serviço
            data = api_client.fetch_prediction(text_input)
            
            # 3. Renderiza os resultados
            st.subheader("Resultados da Classificação")
            predictions = data.get("predictions")
            
            if not predictions:
                st.warning("A API retornou um sucesso, mas sem predições.")
            else:
                for model_name, details in predictions.items():
                    with st.container():
                        render_prediction(model_name, details, data)
                        st.divider()

            st.expander("Ver JSON da resposta completa da API").json(data)

        # 4. Trata erros
        except (APIConnectionError, APIError) as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Um erro inesperado ocorreu na aplicação Streamlit: {e}")