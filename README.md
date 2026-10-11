# College-Assistant-AI-RAG
AI-powered College Assistant built with LangGraph, RAG, FAISS, HuggingFace embeddings, Groq LLM, and Streamlit to answer college-related questions from official PDF documents.


# 🎓 College Assistant AI — RAG-Based College Information Assistant

An AI-powered college information assistant built with **LangGraph, Retrieval-Augmented Generation (RAG), FAISS, HuggingFace Embeddings, Groq LLM, and Streamlit**. It answers college-related questions using information retrieved from official college PDF documents.

The goal is to provide students with a simple conversational interface for accessing academic rules, examination details, course information, fee structures, and college policies—without relying on unsupported answers.

## ✨ Features

- **📚 PDF-Based Knowledge:** Retrieves relevant information from academic handbooks and fee documents.
- **🔎 Semantic Search:** Uses HuggingFace sentence-transformer embeddings and FAISS vector search.
- **🔗 LangGraph Workflow:** Routes questions through classification, retrieval, and response-generation nodes.
- **🤖 LLM-Powered Responses:** Uses Groq-hosted language models to generate natural-language answers.
- **🎯 Domain-Specific Assistant:** Focuses on college academics, examinations, courses, fees, and policies.
- **🛡️ Hallucination Reduction:** Instructs the assistant to answer from retrieved document content and refuse unsupported questions.
- **💬 Streamlit Chat Interface:** Provides an interactive chat UI with programme selection and chat clearing.
- **📄 Source-Aware Answers:** Can include document names and page references when the retrieved PDF metadata supports them.
- **⚙️ Configurable Knowledge Base:** Supports adding or replacing college PDF documents with appropriate code configuration.

## 🧠 How It Works

The application follows a Retrieval-Augmented Generation (RAG) workflow.

1. **Document Loading:** Loads the academic handbook and fee structure PDFs.
2. **Text Splitting:** Splits the extracted text into smaller overlapping chunks.
3. **Embedding Generation:** Converts document chunks into numerical vectors using a sentence-transformer model.
4. **Vector Storage:** Stores the vectors in a FAISS index for similarity search.
5. **Question Classification:** Classifies the user's query as academic, fee-related, a greeting, or out of scope.
6. **Relevant Retrieval:** Searches the appropriate vector database for relevant document chunks.
7. **Context-Grounded Generation:** Sends the question and retrieved context to the LLM.
8. **Response Delivery:** Displays the answer in the Streamlit chat interface or returns a refusal when sufficient supporting information is unavailable.

### Architecture

```text
                 ┌──────────────────────┐
                 │   Student / User     │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Streamlit Chat UI  │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   LangGraph Workflow │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │  Query Classification│
                 └──────────┬───────────┘
                            │
                  ┌─────────┴─────────┐
                  ▼                   ▼
          ┌──────────────┐    ┌──────────────┐
          │ Academic RAG │    │    Fee RAG   │
          └──────┬───────┘    └──────┬───────┘
                 │                   │
                 ▼                   ▼
          ┌──────────────┐    ┌──────────────┐
          │ Academic PDF │    │  Fee PDF     │
          │ FAISS Index  │    │ FAISS Index  │
          └──────┬───────┘    └──────┬───────┘
                 └─────────┬─────────┘
                           ▼
                 ┌──────────────────────┐
                 │ Retrieved Context    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │     Groq LLM         │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Final Answer /       │
                 │ Safe Refusal          │
                 └──────────────────────┘
```

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive chat interface |
| LangGraph | Workflow orchestration and routing |
| LangChain | Document loading, splitting, and LLM integration |
| Groq | Hosted LLM inference |
| HuggingFace Embeddings | Semantic representation of text |
| Sentence Transformers | `all-MiniLM-L6-v2` embedding model |
| FAISS | Vector similarity search |
| PyPDF | PDF document loading |
| python-dotenv | Environment variable management |

## 📂 Project Structure

```text
College-Assistant-AI-RAG/
│
├── app.py
├── academics_handbook.pdf
├── fee_structure.pdf
├── .env
├── college_assistant.py
├── .gitignore
├── requirements.txt
└── README.md
```

The structure above represents the expected basic setup. If your project contains separate modules or folders, update this tree to match your actual repository.

## ⚙️ Prerequisites

Before running the project, install:

- Python 3.10 or a compatible version supported by your installed dependencies.
- pip package manager.
- A Groq API key.
- The required college PDF documents.

The application uses a HuggingFace embedding model. Its first use may download model files, so an internet connection may be required during initial setup.

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/prajapatishubham336/College-Assistant-AI-RAG.git
cd College-Assistant-AI-RAG
```

### 2. Create a virtual environment

**Windows — PowerShell**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt instead:

```bat
venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Replace the placeholder with your actual API key. Never commit the real `.env` file to GitHub.

### 5. Add college documents

Place the required PDF files in the project root:

```text
academics_handbook.pdf
fee_structure.pdf
```

The filenames must match the paths configured in `app.py`.

**Important:** Only distribute these documents if you have permission to share them publicly. If they are confidential or copyrighted, keep them out of the public repository and provide them locally.

### 6. Run the application

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal, usually:

```text
http://localhost:8501
```

Open that URL in your browser.

## 🔑 Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | API key used to access the Groq language model |

Do not hardcode API keys in Python source code or commit secrets to version control.

## 💬 Example Questions

Try the following questions in the chat interface:

**Academic queries**
- What are the examination rules?
- What is the duration of the BCA programme?
- What are the academic policies?

**Fee-related queries**
- What is the fee structure for BBA?
- What are the fees for B.Com (H)?
- What is the fee refund policy?

**Unsupported college information**
- What is the hostel fee?

If the requested information is absent from the indexed documents, the assistant should clearly state that it could not find sufficient information rather than inventing an answer.

**Out-of-scope queries**
- Explain machine learning.
- Write Python code for a calculator.
- Who is the Prime Minister of India?

The assistant should decline these questions because they are outside its configured college-information scope.

*These are test prompts, not guarantees. Actual answers depend on the PDF contents, retrieval quality, classification, and application configuration.*

## 🛡️ Responsible Answering

The application is designed to prioritize document-grounded responses.

- Answers should be supported by retrieved college-document content.
- Missing information should trigger a clear refusal or limitation message.
- Unrelated general questions should be rejected.
- Fees, deadlines, examination rules, and other important information should be verified against the original documents.
- Similarity-search thresholds should be evaluated against representative questions; a fixed threshold alone cannot guarantee factual correctness.

This is a prototype and should not be treated as the final authority for official academic or financial decisions.

## 🧪 Testing Checklist

- [ ] Greeting receives the expected response.
- [ ] Academic questions retrieve relevant handbook content.
- [ ] Fee questions retrieve relevant fee-document content.
- [ ] Unsupported questions receive a safe refusal.
- [ ] Out-of-scope questions are rejected.
- [ ] Programme selection works for BCA, BBA, and B.Com (H).
- [ ] Fee totals match the source documents.
- [ ] Chat clearing works.
- [ ] Missing files and invalid API credentials produce useful error messages.
- [ ] Answers do not invent unsupported fees, policies, or dates.

## ⚠️ Limitations

- The assistant can only answer reliably when relevant information is present in the indexed documents.
- Incorrect, incomplete, or outdated PDFs can lead to incomplete or outdated answers.
- Query classification and semantic retrieval may occasionally select irrelevant content.
- The first model download and document indexing can take additional time.
- Access to Groq depends on the availability and terms of the configured model and API service.
- This version is not a replacement for official college administration or student support.

## 🔮 Future Improvements

- Add support for more college documents and document collections.
- Improve retrieval with metadata filtering and hybrid search.
- Add stronger answer-grounding and relevance validation.
- Display citations with document name and page number for every supported answer.
- Add follow-up question handling and conversation-aware retrieval.
- Add automated unit and integration tests.
- Add deployment configuration and secure secret management.
- Add document upload and index-refresh functionality.
- Add multilingual support for English and Hinglish queries.

## 🤝 Contributing

Contributions and suggestions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Make your changes and test them.
4. Submit a pull request describing your improvements.

Please do not include API keys, private student information, or confidential college documents in commits.

## 🔒 Security

- Keep `.env` and all API credentials private.
- Review PDF distribution permissions before publishing documents.
- Do not upload private student records or sensitive institutional data.
- Rotate exposed API keys immediately.

## 👨‍💻 Author

**Shubham Prajapati**

AI/ML Engineer | Machine Learning | Generative AI

GitHub: [@prajapatishubham336](https://github.com/prajapatishubham336)

## 📄 License

Choose a license before making this repository public. For example, the MIT License is suitable for many open-source software projects, provided you have the right to license all included code and assets.

If you choose MIT, add a `LICENSE` file and confirm that all third-party materials and college documents may legally be distributed under your repository's terms.

---

**Built with Python, LangGraph, RAG, FAISS, HuggingFace Embeddings, Groq, and Streamlit.**

*Making college information easier to access through document-grounded AI.*
