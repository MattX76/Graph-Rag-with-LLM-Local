# 📚 Local RAG Chatbot con Knowledge Graphs

Asistente conversacional que funciona completamente de forma local, combinando **modelos de lenguaje** y **búsqueda inteligente** para ofrecer respuestas precisas basadas en tus documentos.  
Integra búsqueda híbrida (base estructural + FAISS (base vectorial para busqueda semantica)), construcción de grafos de conocimiento y un sistema de memoria para mantener el contexto.  
Soporta la carga de archivos PDF, DOCX y TXT, procesando la información sin depender de servidores externos.

---

## 🔧 Requisitos Previos

Antes de empezar, asegúrate de tener:  

- 🐍 **Python 3.10+**  
- 📦 **pip** actualizado  
- 🤖 **Ollama** para la gestión de modelos locales  
- 🐳 (Opcional) **Docker** para contenedores  

---

## 🛠 Instalación Paso a Paso

### 1️⃣ Instalación Manual con Python y Entorno Virtual

```bash
git clone https://github.com/MattX76/Graph-Rag-with-LLM-Local

# Crear entorno virtual
python -m venv venv

# Activar entorno
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Actualizar pip (opcional)
pip install --upgrade pip

# Instalar dependencias
pip install -r requirements.txt
```
### 2️⃣ Configuración de Ollama
Descargar e instalar Ollama → https://ollama.com/

Descargar modelos necesarios:
```bash
ollama pull deepseek-r1:7b
ollama pull nomic-embed-text
```
🔧 Nota: Para otros modelos, modifica MODEL o EMBEDDINGS_MODEL en .env.

### 3️⃣ Ejecución de la Aplicación
```bash
ollama serve
streamlit run app.py
```
🌐 Abre tu navegador en http://localhost:xxxx

### 🐳 Instalación con Docker
Opción A: Usar Ollama desde tu máquina (host)
```bash
docker-compose build
docker-compose up
Opción B: Todo en contenedores (Ollama + Chatbot)
```
```yaml
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
```
```bash
docker-compose build
docker-compose up
```
### 🌐 La aplicación estará disponible en http://localhost:8501

🧠 Cómo Funciona
📄 Subida de documentos (PDF, DOCX, TXT)

🔍 Recuperación híbrida: BM25 + FAISS localizan fragmentos relevantes

🧩 GraphRAG: crea un grafo de conocimiento para entender contexto y relaciones

🧠 Reordenamiento neuronal con Cross-Encoder

💡 Expansión de consultas (HyDE) para mayor precisión

🗂 Memoria conversacional para mantener el hilo del diálogo

🤖 Generación final con el modelo seleccionado en Ollama
