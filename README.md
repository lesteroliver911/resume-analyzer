# Resume Analyzer

An AI-powered application that analyzes resumes against job descriptions using LangChain, OpenAI, and LlamaParse.

## Features
- Multiple analysis modes (single JD vs multiple resumes, multiple JDs vs single resume, batch processing)
- AI-powered matching and analysis
- Vector store-based document search
- Conversation memory for context-aware responses

## Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your API keys
4. Run the application: `python main.py`

## Requirements
- Python 3.10+
- OpenAI API key
- LlamaParse API key

## Directory Structure
- `/templates`: HTML templates
- `/static`: CSS and other static files
- `/uploads`: Temporary upload directory (git-ignored)
- `/parsed_documents`: Processed documents (git-ignored)
- `/faiss_index`: Vector store indices (git-ignored)
