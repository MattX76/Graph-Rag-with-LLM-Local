# 📚 Local RAG Chatbot con Knowledge Graphs

Asistente conversacional que funciona completamente de forma local, combinando **modelos de lenguaje** y **búsqueda inteligente** para ofrecer respuestas precisas basadas en tus documentos.  
Integra búsqueda híbrida (base estructural + FAISS (base vectorial para busqueda semantica)), construcción de grafos de conocimiento y un sistema de memoria para mantener el contexto.  
Soporta la carga de archivos PDF, DOCX y TXT, procesando la información sin depender de servidores externos.

---

## 🔧 Requisitos Previos

Antes de comenzar, asegúrate de contar con:
- **Python 3.10+** instalado en tu sistema.
- **pip** actualizado.
- **Ollama** instalado para la gestión de modelos locales.
- (Opcional) **Docker** si prefieres una implementación en contenedores.

---

## 🛠 Instalación Paso a Paso

### 1️⃣ Instalación Manual con Python y Entorno Virtual
```bash
git clone 
cd Local-RAG-KG-Chatbot

# Crear entorno virtual
python -m venv venv

# Activar entorno
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate

# (Opcional) Actualizar pip
pip install --upgrade pip

# Instalar dependencias
pip install -r requirements.txt
2️⃣ Configuración de Ollama
Descarga e instala Ollama → https://ollama.com/

Descarga los modelos necesarios:

```bash
Copiar
Editar
ollama pull deepseek-r1:7b
ollama pull nomic-embed-text
Si quieres utilizar otros modelos, modifica las variables MODEL o EMBEDDINGS_MODEL en el archivo .env.

### 3️⃣ Ejecución de la Aplicación
```bash
ollama serve
streamlit run app.py
```
Abre tu navegador en http://localhost:xxxx para interactuar con el asistente.

🐳 Instalación con Docker
Opción A: Usar Ollama desde tu máquina (host)
```bash
Copiar
Editar
docker-compose build
docker-compose up
La aplicación se abrirá en http://localhost:8501.

Opción B: Todo en contenedores (Ollama + Chatbot)
yaml
Copiar
Editar
version: "3.8"

services:
  ollama:
    image: ghcr.io/jmorganca/ollama:latest
    container_name: ollama
    ports:
      - "11434:11434"

  rag-chatbot:
    container_name: rag-chatbot
    build: .
    ports:
      - "8501:8501"
    environment:
      - OLLAMA_API_URL=http://ollama:11434
      - MODEL=deepseek-r1:7b
      - EMBEDDINGS_MODEL=nomic-embed-text:latest
      - CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
    depends_on:
      - ollama
Luego ejecuta:

docker-compose build
docker-compose up
🧠 Cómo Funciona
Sube tus documentos (PDF, DOCX, TXT).

Recuperación híbrida: BM25 y FAISS localizan los fragmentos más relevantes.

GraphRAG: crea un grafo de conocimiento para entender relaciones y contexto.

Reordenamiento neuronal con Cross-Encoder para priorizar resultados.

Expansión de consultas (HyDE) para mejorar la precisión.

Memoria conversacional para mantener el hilo del diálogo.

Generación final con el modelo seleccionado en Ollama.
