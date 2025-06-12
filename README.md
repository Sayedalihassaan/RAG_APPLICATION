# 🤖 RAG-based Document QA App using LangChain & Gemini

This project is a **Retrieval-Augmented Generation (RAG)** application built with **Streamlit**, **LangChain**, and **Google's Gemini API**. It allows users to upload documents (PDF, DOCX, or TXT), process them into chunks, embed them using **Google Generative AI Embeddings**, and interact with the content via conversational or standard Q&A.

## 🚀 Features

- 📄 Upload and process `.pdf`, `.docx`, and `.txt` files.
- 🔍 Ask questions directly from your documents.
- 🧠 Optional conversational memory for follow-up Q&A.
- ⚡ Uses **FAISS** for fast vector-based retrieval.
- 🌐 Powered by **Google Gemini** (via `langchain-google-genai`).
- 🛠️ Full configuration from the Streamlit sidebar.

---

## 📸 Demo

![App Screenshot](./assets/download.png)

---

## 🧰 Tech Stack

- [LangChain](https://www.langchain.com/)
- [Streamlit](https://streamlit.io/)
- [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)
- [Google Gemini + Embeddings](https://makersuite.google.com/app)
- [dotenv](https://pypi.org/project/python-dotenv/)

---

## 📂 Project Structure

```

.
├── assets/
│   └── download.png          # Optional logo or image
├── .env                      # Store your Google API key securely
├── APP.py                   # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # You're reading it!
└── create\_structure.py       # Script to initialize project files

````

---

## 🔧 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/rag-doc-qa-app.git
cd rag-doc-qa-app
````

### 2. Create virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your API key

Create a `.env` file and add:

```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

### 5. Run the app

```bash
streamlit run APP.py
```

---

## 📁 Process Flow

1. Upload a document (PDF, DOCX, or TXT).
2. The document is chunked using `RecursiveCharacterTextSplitter`.
3. Chunks are embedded using **Google Generative AI Embeddings**.
4. Vectors are stored in a local **FAISS** index.
5. User inputs questions in the UI.
6. The app retrieves relevant chunks and sends them to **Gemini LLM** for response.
7. Optionally use memory for contextual follow-up questions.

---

## 🛡️ API Note

This project uses **Google Gemini (Generative AI)** via `langchain-google-genai`. You can obtain an API key from [Google AI Studio](https://makersuite.google.com/app) and manage limits from your Google Cloud Console.

---

## 🧪 Example Use Cases

* Reading and querying long research papers
* Company policy document Q\&A
* Chat-style exploration of legal or technical documents

---

## 📌 TODO

* ✅ Multi-format document ingestion
* ✅ Embedding with Gemini
* ✅ FAISS vector storage
* ✅ Memory-enabled chat
* 🔜 PDF preview
* 🔜 File management dashboard
* 🔜 Export chat logs

---

## 👨‍💻 Author

**Sayed Ali**
AI Engineer | Python Enthusiast | Document Intelligence
[LinkedIn](https://www.linkedin.com/in/sayed-ali-482668262/) • [GitHub](https://github.com/Sayedalihassaan)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

* [LangChain](https://www.langchain.com/)
* [Google Gemini API](https://ai.google.dev/)
* [FAISS](https://github.com/facebookresearch/faiss)
* [Streamlit](https://streamlit.io/)

```
