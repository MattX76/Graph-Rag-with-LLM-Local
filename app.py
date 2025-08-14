# app.py — Versión Hugging Face (sin Ollama), en español

import os
import pickle
from threading import Thread

import streamlit as st
import torch
import faiss
from dotenv import load_dotenv, find_dotenv
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# =========================
# Carga .env y configuración de dispositivo
# =========================
load_dotenv(find_dotenv())
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Configuración (variables de entorno)
# =========================
MODEL_HF = os.getenv("MODEL_HF", "Qwen/Qwen2.5-7B-Instruct")
EMBEDDINGS_MODEL_HF = os.getenv("EMBEDDINGS_MODEL_HF", "BAAI/bge-m3")
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-Multilingual-MiniLM-L12-v2")
INDEX_DIR = os.getenv("INDEX_DIR", "faiss_store")
os.makedirs(INDEX_DIR, exist_ok=True)

# Fix heredado del repo original (inofensivo si falla)
try:
    torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]
except Exception:
    pass

# =========================
# Recursos cacheados (se cargan una sola vez por sesión)
# =========================
@st.cache_resource
def get_embedder() -> SentenceTransformer:
    # Embeddings con Sentence-Transformers (multilingüe, recomendado bge-m3)
    return SentenceTransformer(EMBEDDINGS_MODEL_HF, device=device)

@st.cache_resource
def get_llm():
    """
    Devuelve (tokenizer, model) listos para generación.
    En GPU A100 usa bfloat16 automáticamente; en CPU, float32.
    """
    tok = AutoTokenizer.from_pretrained(MODEL_HF, use_fast=True)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_HF,
        torch_dtype=dtype if device == "cuda" else None,
        device_map="auto" if device == "cuda" else None,
    )
    return tok, model

@st.cache_resource
def get_reranker():
    try:
        return CrossEncoder(CROSS_ENCODER_MODEL, device=device)
    except Exception as e:
        st.warning(f"No se pudo cargar el modelo de reranking: {e}")
        return None

# =========================
# Estado de sesión
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False

# =========================
# UI / Estilos
# =========================
st.set_page_config(page_title="DeepGraph RAG-Pro (HF)", layout="wide")
st.markdown("""
    <style>
        .stApp { background-color: #f4f4f9; }
        h1 { color: #00B894; text-align: center; }
        .stChatMessage { border-radius: 10px; padding: 10px; margin: 10px 0; }
        .stChatMessage.user { background-color: #e8f0fe; }
        .stChatMessage.assistant { background-color: #d1e7dd; }
        .stButton>button { background-color: #0984e3; color: white; }
    </style>
""", unsafe_allow_html=True)

# =========================
# Barra lateral
# =========================
with st.sidebar:
    st.header("📁 Gestión de documentos")
    uploaded_files = st.file_uploader(
        "Sube documentos (PDF / DOCX / TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    st.markdown("---")
    st.header("⚙️ Configuración RAG")
    st.session_state.rag_enabled = st.checkbox("Activar RAG", value=True)
    st.session_state.enable_hyde = st.checkbox("Activar HyDE (expansión de consulta)", value=True)
    st.session_state.enable_reranking = st.checkbox("Activar reranking neuronal", value=True)
    st.session_state.enable_graph_rag = st.checkbox("Activar GraphRAG (placeholder)", value=False)

    st.session_state.temperature = st.slider("Temperatura", 0.0, 1.0, 0.3, 0.05)
    st.session_state.top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
    st.session_state.max_contexts = st.slider("Contextos a usar", 1, 5, 3)
    st.session_state.max_new_tokens = st.slider("Tokens de salida máx.", 64, 2048, 400, 32)

    # Construir / actualizar índice FAISS con HF
    if st.button("Construir / Actualizar índice"):
        if not uploaded_files:
            st.warning("Primero sube documentos.")
        else:
            with st.spinner("Procesando documentos y construyendo el índice..."):
                # Implementación HF en utils (la actualizamos en el siguiente paso)
                from utils.doc_handler import process_documents
                process_documents(
                    uploaded_files=uploaded_files,
                    embedder=get_embedder(),
                    index_dir=INDEX_DIR
                )
                st.session_state.documents_loaded = True
                st.session_state.index_ready = True
            st.success("Índice FAISS construido/actualizado.")

    if st.button("Borrar historial de chat"):
        st.session_state.messages = []
        st.rerun()

    st.sidebar.markdown("""
        <div style="position: absolute; top: 20px; right: 10px; font-size: 12px; color: gray;">
            <b>Desarrollado por:</b> N Sai Akhil &copy; Todos los derechos reservados 2025
        </div>
    """, unsafe_allow_html=True)

# =========================
# Cabecera
# =========================
st.title("🤖 DeepGraph RAG-Pro (Hugging Face)")
st.caption("RAG con FAISS + Embeddings HF + Reranking + HyDE (opcional) — sin Ollama")

# Mostrar historial de mensajes
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================
# Bucle de chat
# =========================
if prompt := st.chat_input("Pregunta sobre tus documentos..."):
    # Últimas 5 interacciones para mantener contexto
    chat_history = "\n".join([m["content"] for m in st.session_state.messages[-5:]])
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        respuesta_placeholder = st.empty()
        respuesta_completa = ""

        # ====== 1) Recuperación de contexto ======
        contexto = ""
        pasajes = []
        if st.session_state.rag_enabled and st.session_state.index_ready:
            try:
                # Implementación HF en utils (la actualizamos en el siguiente paso)
                from utils.retriever_pipeline import retrieve_documents
                pasajes = retrieve_documents(
                    query=prompt,
                    index_dir=INDEX_DIR,
                    embedder=get_embedder(),
                    reranker=get_reranker() if st.session_state.enable_reranking else None,
                    chat_history=chat_history,
                    enable_hyde=st.session_state.enable_hyde,
                    k=12,
                    topk=st.session_state.max_contexts
                )
                contexto = "\n".join([f"[Fuente {i+1}]: {p['page_content']}" for i, p in enumerate(pasajes)])
            except Exception as e:
                st.error(f"Error en recuperación: {e}")

        # ====== 2) Construcción del prompt ======
        system_prompt = f"""Usa el historial de conversación para mantener el contexto.
Historial de chat:
{chat_history}

Analiza la pregunta y el contexto siguiendo estos pasos:
1) Identifica entidades y relaciones clave.
2) Verifica si hay contradicciones entre las fuentes.
3) Sintetiza la información de múltiples fragmentos.
4) Redacta una respuesta clara y estructurada en español, citando [1], [2]... cuando uses el contexto.

Contexto:
{contexto}

Pregunta: {prompt}
Respuesta:
"""

        # ====== 3) Generación con Hugging Face (streaming) ======
        tok, model = get_llm()
        inputs = tok(system_prompt, return_tensors="pt").to(model.device)
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=int(st.session_state.max_new_tokens),
            do_sample=st.session_state.temperature > 0,
            temperature=float(st.session_state.temperature),
            top_p=float(st.session_state.top_p),
            streamer=streamer,
            pad_token_id=tok.eos_token_id,
        )

        hilo = Thread(target=model.generate, kwargs=gen_kwargs)
        hilo.start()
        try:
            for token in streamer:
                respuesta_completa += token
                respuesta_placeholder.markdown(respuesta_completa + "▌")
        except Exception as e:
            st.error(f"Error en generación: {e}")

        respuesta_placeholder.markdown(respuesta_completa)
        st.session_state.messages.append({"role": "assistant", "content": respuesta_completa})
