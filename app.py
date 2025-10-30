from dotenv import load_dotenv
import os
import traceback
import fitz  # PyMuPDF
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# LangChain components
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# --- Initialize Embeddings (Local-Only Version) ---
embeddings = None
embedding_type = "none"

print("=" * 50)
print("Initializing Embeddings...")
print("=" * 50)

# Always try HuggingFace first (no API or quota needed)
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("Attempting HuggingFace embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    test_embed = embeddings.embed_query("test")
    embedding_type = "HuggingFace"
    print(f"✓ HuggingFace embeddings loaded successfully! (dim: {len(test_embed)})")

except Exception as e:
    print(f"✗ HuggingFace failed: {e}")
    print("Using fallback: Custom TF-IDF embeddings")
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    from langchain.embeddings.base import Embeddings

    class TFIDFEmbeddings(Embeddings):
        def __init__(self):
            self.vectorizer = TfidfVectorizer(max_features=384)
            self.fitted = False

        def fit(self, texts):
            self.vectorizer.fit(texts)
            self.fitted = True

        def embed_documents(self, texts):
            if not self.fitted:
                self.fit(texts)
            vectors = self.vectorizer.transform(texts).toarray()
            return [self._pad(v) for v in vectors]

        def embed_query(self, text):
            if not self.fitted:
                return [0.0] * 384
            vec = self.vectorizer.transform([text]).toarray()[0]
            return self._pad(vec)

        def _pad(self, vec):
            if len(vec) < 384:
                vec = np.pad(vec, (0, 384 - len(vec)))
            else:
                vec = vec[:384]
            return vec.tolist()

    embeddings = TFIDFEmbeddings()
    embedding_type = "TF-IDF (Fallback)"
    print("✓ TF-IDF embeddings initialized")

print(f"Embedding type: {embedding_type}")
print("=" * 50)

# --- Flask App Configuration ---
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ FIXED PATH for Render — frontend folder inside the repo
app = Flask(__name__, static_folder=os.path.join(APP_ROOT, "frontend"), static_url_path="")
CORS(app)

# Global vector store
db = None
current_file = None


def extract_text_from_pdf(path):
    """Extract text from PDF using PyMuPDF."""
    text = []
    with fitz.open(path) as doc:
        for page in doc:
            text.append(page.get_text())
    return "\n".join(text)


# ✅ Serve frontend
@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')


# ✅ Handle all other routes for frontend navigation (important for Vercel/Render)
@app.errorhandler(404)
def not_found(e):
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    if request.method == 'OPTIONS':
        return '', 204

    global db, current_file

    try:
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file provided"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        print(f"✓ File saved: {filename}")

        # Extract text
        if filename.lower().endswith('.pdf'):
            print("Extracting PDF text...")
            text_data = extract_text_from_pdf(filepath)
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                text_data = f.read()

        if not text_data.strip():
            return jsonify({"error": "No extractable text"}), 400

        print(f"✓ Extracted {len(text_data)} characters")

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text_data)
        print(f"✓ Created {len(chunks)} chunks")

        # Fit TF-IDF if using fallback
        if embedding_type == "TF-IDF (Fallback)":
            print("Fitting TF-IDF vectorizer...")
            embeddings.fit(chunks)

        documents = [Document(page_content=chunk, metadata={"chunk": i})
                     for i, chunk in enumerate(chunks)]

        print("Building FAISS index...")
        db = FAISS.from_documents(documents, embeddings)
        current_file = filename
        print("✓ FAISS index created successfully!")

        return jsonify({
            "message": f"✓ File '{filename}' uploaded successfully!",
            "chunks": len(chunks),
            "characters": len(text_data),
            "embedding_type": embedding_type
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/ask', methods=['POST', 'OPTIONS'])
def ask_question():
    if request.method == 'OPTIONS':
        return '', 204

    global db
    if db is None:
        return jsonify({"answer": "⚠️ Please upload a document first"}), 400

    try:
        data = request.get_json() or {}
        question = data.get('question', '').strip()

        if not question:
            return jsonify({"answer": "Please provide a question"}), 400

        print(f"Question: {question}")
        docs = db.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        print(f"✓ Found {len(docs)} relevant chunks")

        prompt_template = """Use the following context to answer the question. 
If the context does not contain enough information, say you cannot answer.

Context:
{context}

Question: {question}

Answer:"""

        print("Initializing Groq LLM...")
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=500,
            api_key=os.getenv("GROQ_API_KEY")
        )

        print("Generating answer...")
        prompt = prompt_template.format(context=context, question=question)
        response = llm.invoke(prompt)
        answer = getattr(response, "content", str(response))

        print(f"✓ Answer generated ({len(answer)} chars)")
        return jsonify({"answer": answer})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"answer": f"Error: {str(e)}"}), 500


@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "db_loaded": db is not None,
        "current_file": current_file,
        "embedding_type": embedding_type,
        "groq_key_set": bool(os.getenv("GROQ_API_KEY"))
    })


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("STARTING RAG CHAT APP")
    print("=" * 50)
    print(f"Groq API Key: {'✓ Set' if os.getenv('GROQ_API_KEY') else '✗ Not set'}")
    print(f"Embedding Type: {embedding_type}")
    print("=" * 50 + "\n")
    app.run(host='0.0.0.0', port=10000)  # ✅ Render runs on port 10000
