from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# Import your existing components
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

# Initialize FastAPI app
app = FastAPI(title="TOA Personal Copilot API")

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize your LLM and chain
model = OllamaLLM(model="tinyllama")

template = """
You are an expert programming assistant. Use the provided context to answer the question accurately.

Relevant Context: {context}

Question: {question}

Provide a helpful and detailed answer based on the context. If the context doesn't contain the answer, use your general knowledge but mention that you're doing so.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    sources: List[str] = []

# Routes
@app.get("/")
async def root():
    return {"message": "TOA Personal Copilot API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "llama3.2"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Get user message from request
        user_message = request.message
        
        # Retrieve relevant documents using your existing retriever
        relevant_docs = retriever.invoke(user_message)
        
        # Extract content from documents for context
        context_content = [doc.page_content for doc in relevant_docs]
        
        # Generate response using your chain
        result = chain.invoke({
            "context": context_content, 
            "question": user_message
        })
        
        # Extract sources from metadata if available
        sources = []
        for doc in relevant_docs:
            if hasattr(doc, 'metadata'):
                # Try different possible metadata fields
                if 'title' in doc.metadata:
                    sources.append(doc.metadata['title'])
                elif 'category' in doc.metadata:
                    sources.append(doc.metadata['category'])
                else:
                    sources.append("Document")
            else:
                sources.append("Unknown source")
        
        return ChatResponse(
            response=result,
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/upload")
async def upload_file():
    return {"message": "File upload endpoint - to be implemented"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)