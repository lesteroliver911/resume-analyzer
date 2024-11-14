import os
import logging
from flask import Flask, render_template, request, session, jsonify
from dotenv import load_dotenv
from llama_parse import LlamaParse
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.agents import AgentExecutor, Tool, create_openai_functions_agent
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Set a secret key for session management

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Ensure you have set your API keys in the environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

if not OPENAI_API_KEY or not LLAMA_CLOUD_API_KEY:
    raise ValueError("API keys not found in environment variables")

# Initialize LlamaParse
parser = LlamaParse(
    api_key=LLAMA_CLOUD_API_KEY,
    result_type="markdown",
    use_vendor_multimodal_model=True,
    vendor_multimodal_model_name="openai-gpt-4o-mini",
    verbose=True
)

# Initialize OpenAI components
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.1)

# Ensure necessary directories exist
OUTPUT_FOLDER = 'parsed_documents'
FAISS_INDEX_FOLDER = './faiss_index'
UPLOAD_FOLDER = 'uploads'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(FAISS_INDEX_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def parse_document(file, output_name: str):
    try:
        logger.info(f"Parsing document: {output_name}")
        file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
        file.save(file_path)
        result = parser.load_data(file_path, extra_info={"file_name": file.filename})
        output_path = os.path.join(OUTPUT_FOLDER, f"{output_name}.md")
        with open(output_path, 'w', encoding='utf-8') as f:
            for page in result:
                f.write(page.text)
        logger.info(f"Document parsed successfully: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error parsing {output_name}: {str(e)}")
        return None

def create_vector_store(file_path: str, collection_name: str):
    try:
        logger.info(f"Creating vector store for: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        vectorstore = FAISS.from_texts([text], embeddings)
        save_path = os.path.join(FAISS_INDEX_FOLDER, collection_name)
        vectorstore.save_local(save_path)
        logger.info(f"Vector store created: {save_path}")
    except Exception as e:
        logger.error(f"Error creating vector store for {file_path}: {str(e)}")
        raise

@tool("search_documents")
def search_documents(query: str, collection_name: str):
    """Search for information in the documents using the query."""
    try:
        logger.info(f"Searching documents: {collection_name}")
        vectorstore = FAISS.load_local(os.path.join(FAISS_INDEX_FOLDER, collection_name), embeddings)
        docs = vectorstore.similarity_search(query, k=1)
        return docs[0].page_content if docs else "No relevant information found."
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        return f"Error searching documents: {str(e)}"

@tool("analyze_match")
def analyze_match(content: str):
    """Analyze the match between job requirements and the candidate's resume."""
    try:
        logger.info("Analyzing match between job description and resume")
        # Split the content into job description and resume
        parts = content.split("Resume:")
        if len(parts) != 2:
            return "Error: Invalid input format. Expected 'Job Description:' followed by 'Resume:'"
        
        jd_content = parts[0].strip()
        resume_content = parts[1].strip()
        
        analysis_prompt = f"""
        Analyze the match between the job requirements and the candidate's resume:
        
        Job Description:
        {jd_content}
        
        Candidate's Resume:
        {resume_content}
        
        Provide:
        1. Match Percentage: Estimate how well the candidate matches the job requirements (0-100%).
        2. Key Strengths: Identify the candidate's main strengths relevant to the job.
        3. Potential Gaps: Note any significant mismatches or missing skills.
        4. Overall Assessment: Briefly summarize the candidate's suitability for the role.
        """
        return llm.invoke(analysis_prompt).content
    except Exception as e:
        logger.error(f"Error analyzing match: {str(e)}")
        return f"Error analyzing match: {str(e)}"

def setup_agent():
    tools = [
        Tool(name="search_documents", func=search_documents, description="Search for information in documents."),
        Tool(name="analyze_match", func=analyze_match, description="Analyze the match between job and resume.")
    ]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI assistant specialized in analyzing job descriptions and resumes. Your task is to provide detailed, well-structured, and insightful comparisons between job requirements and candidate qualifications. Follow these guidelines:

1. Thoroughly examine both the job description and resume.
2. Provide a clear, percentage-based match score (0-100%).
3. Identify and list key strengths of the candidate that align with the job requirements.
4. Highlight any potential gaps or mismatches between the job needs and candidate's profile.
5. Offer a concise overall assessment of the candidate's suitability for the role.
6. Use bullet points and clear headings to structure your response.
7. Be objective and base your analysis solely on the provided information.
8. If you need more information, use the search_documents tool to find relevant details.
9. Use the analyze_match tool for in-depth comparisons when necessary.

Your response should be well-formatted html with inline css."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = create_openai_functions_agent(llm, tools, prompt)
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True
    )

def process_files(jd_files, resume_files):
    agent_executor = setup_agent()
    results = []

    for jd_file in jd_files:
        parsed_jd_path = parse_document(jd_file, f'jd_{jd_file.filename}')
        if not parsed_jd_path:
            logger.error(f"Failed to parse job description: {jd_file.filename}")
            continue
        create_vector_store(parsed_jd_path, f'jd_collection_{jd_file.filename}')

        with open(parsed_jd_path, 'r', encoding='utf-8') as f:
            jd_content = f.read()

        for resume_file in resume_files:
            parsed_resume_path = parse_document(resume_file, f'resume_{resume_file.filename}')
            if not parsed_resume_path:
                logger.error(f"Failed to parse resume: {resume_file.filename}")
                continue
            create_vector_store(parsed_resume_path, f'resume_collection_{resume_file.filename}')

            with open(parsed_resume_path, 'r', encoding='utf-8') as f:
                resume_content = f.read()

            analysis_query = f"""Analyze the match between this job description and resume:
            Job Description: {jd_content}
            Resume: {resume_content}"""

            response = agent_executor.invoke({"input": analysis_query})

            results.append({
                "job_description": jd_file.filename,
                "resume": resume_file.filename,
                "analysis": response['output']
            })

    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    analysis_mode = request.form['analysis_mode']
    
    if analysis_mode == "single_jd_multiple_resumes":
        jd_file = request.files['jd_file']
        resume_files = request.files.getlist('resume_files')
        results = process_files([jd_file], resume_files)
    elif analysis_mode == "multiple_jds_single_resume":
        jd_files = request.files.getlist('jd_files')
        resume_file = request.files['resume_file']
        results = process_files(jd_files, [resume_file])
    elif analysis_mode == "batch_processing":
        jd_files = request.files.getlist('batch_jd_files')
        resume_files = request.files.getlist('batch_resume_files')
        results = process_files(jd_files, resume_files)
    else:
        return jsonify({"error": "Invalid analysis mode"}), 400
    
    return jsonify({"results": results})

if __name__ == '__main__':
    app.run(debug=True)
