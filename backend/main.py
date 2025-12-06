from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
from datetime import datetime
import os
import tempfile
import wave
import numpy as np
from scipy import signal

# Import SQLAlchemy components
from sqlalchemy.orm import Session

# Import your existing components
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever, add_documents, search_documents

# Import new components
from pdf_processor import extract_text_from_pdf, extract_text_from_image
from database import init_db, get_db_dep
from repositories import ConversationRepository, MessageRepository
from models import Conversation as ConversationModel, Message as MessageModel
from hallucination_check import HallucinationChecker, get_response_quality_metrics

# Try to import whisper for speech-to-text, fall back if not available
try:
    import whisper
    WHISPER_AVAILABLE = True
    print("✓ Whisper module loaded successfully")
except ImportError as e:
    WHISPER_AVAILABLE = False
    print(f"✗ Whisper not available: {e}")
    print("   To fix: pip install openai-whisper")

# Initialize FastAPI app
app = FastAPI(title="TOA Personal Copilot API")

# Initialize database
init_db()

# Initialize hallucination checker
hallucination_checker = HallucinationChecker(threshold=0.3)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize your LLM and chain
model = OllamaLLM(model="llama3.2:3b")

template = """
You are a helpful, friendly, and knowledgeable AI assistant. You help users with a wide variety of topics including programming, general knowledge, explanations, creative writing, and more.

Your responsibilities:
1. When the user asks about documents/images they've uploaded, prioritize information from those documents
2. For general questions, provide accurate and helpful answers using your general knowledge
3. Be conversational and engaging while maintaining accuracy
4. Keep track of the conversation context to provide coherent responses
5. If you don't know something, be honest about it

Conversation History (for context):
{chat_history}

Available Document/Image Information:
{context}

User Question: {question}

Please provide a helpful, accurate, and conversational response. If you're using information from uploaded documents, reference them naturally.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    sources: List[str] = []
    provenance: Optional[Dict[str, Any]] = None
    hallucination_score: Optional[float] = None
    is_grounded: Optional[bool] = None

class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: Optional[int] = 0

class MessageResponse(BaseModel):
    id: int
    conversation_id: str
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime

class TranscribeResponse(BaseModel):
    text: str

# Routes
@app.get("/")
async def root():
    return {"message": "TOA Personal Copilot API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "llama3.2", "whisper_available": WHISPER_AVAILABLE}

@app.get("/conversations", response_model=List[ConversationResponse])
async def get_conversations(db: Session = Depends(get_db_dep)):
    """Get all conversations with message count"""
    conversations = ConversationRepository.get_all(db)
    
    response = []
    for conv in conversations:
        # Get message count
        message_count = len(conv.messages)
        response.append(ConversationResponse(
            id=conv.id,
            title=conv.title,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            message_count=message_count
        ))
    
    return response

@app.post("/conversations", response_model=ConversationResponse)
async def create_new_conversation(db: Session = Depends(get_db_dep)):
    """Create a new conversation"""
    conversation = ConversationRepository.create(db)
    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        message_count=0
    )

@app.get("/conversations/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_messages(conversation_id: str, db: Session = Depends(get_db_dep)):
    """Get all messages for a conversation"""
    messages = MessageRepository.get_by_conversation(db, conversation_id)
    
    return [
        MessageResponse(
            id=msg.id,
            conversation_id=msg.conversation_id,
            role=msg.role,
            content=msg.content,
            metadata=msg.message_metadata,
            created_at=msg.created_at
        )
        for msg in messages
    ]

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, db: Session = Depends(get_db_dep)):
    """Delete a conversation"""
    success = ConversationRepository.delete(db, conversation_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "deleted"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest, 
    db: Session = Depends(get_db_dep)
):
    """
    Handle chat messages with SOTA hybrid retrieval, hallucination detection,
    and page-level provenance tracking.
    """
    try:
        # Create new conversation if needed
        if not request.conversation_id:
            conversation = ConversationRepository.create(db)
            db.commit()
            conversation_id = conversation.id
        else:
            # Verify conversation exists
            conversation = ConversationRepository.get_by_id(db, request.conversation_id)
            if not conversation:
                raise HTTPException(status_code=404, detail="Conversation not found")
            conversation_id = request.conversation_id
            # Update conversation timestamp
            ConversationRepository.update_timestamp(db, conversation_id)
            db.commit()
        
        # Save user message
        user_message = MessageRepository.create(
            db=db,
            conversation_id=conversation_id,
            role="user",
            content=request.message
        )
        db.commit()
        
        # Get last 10 messages for context (excluding current)
        recent_messages = MessageRepository.get_last_n_messages(db, conversation_id, n=10)
        
        # Format chat history - include full content for uploaded files (system messages)
        chat_history_parts = []
        for msg in recent_messages:
            if msg.id == user_message.id:
                continue
            
            # For file uploads (system messages), include full content
            if msg.role == "system":
                chat_history_parts.append(f"[SYSTEM] {msg.content}")
            else:
                # For regular messages, use preview
                content = msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
                chat_history_parts.append(f"{msg.role}: {content}")
        
        chat_history = "\n".join(chat_history_parts) if chat_history_parts else ""
        
        # ===== STAGE 1: HYBRID RETRIEVAL (BM25 + Vector + Cross-Encoder Reranking) =====
        print(f"🔍 Searching with hybrid retrieval for: '{request.message[:100]}'")
        relevant_docs = search_documents(conversation_id, request.message, k=5)

        # Extract context and join into single string
        context_content = "\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else "No uploaded documents available."
        
        print(f"✓ Retrieved {len(relevant_docs)} documents with hybrid search + reranking")
        
        # Generate response
        result = chain.invoke({
            "chat_history": chat_history,
            "context": context_content, 
            "question": request.message
        })
        
        # ===== STAGE 2: HALLUCINATION DETECTION & PROVENANCE =====
        quality_metrics = get_response_quality_metrics(result, relevant_docs, hallucination_checker)
        
        hallucination_check = quality_metrics['hallucination']
        provenance = quality_metrics['provenance']
        
        print(f"📊 Hallucination Score: {hallucination_check['hallucination_score']} | "
              f"Confidence: {hallucination_check['confidence']} | "
              f"Is Reliable: {quality_metrics['is_reliable']}")
        
        # Add quality metrics to response metadata
        response_metadata = {
            'hallucination_score': hallucination_check['hallucination_score'],
            'confidence': hallucination_check['confidence'],
            'is_grounded': not hallucination_check['is_hallucinating'],
            'sources_count': len(relevant_docs),
            'grounded_statements': len(hallucination_check.get('grounded_statements', [])),
            'ungrounded_statements': len(hallucination_check.get('ungrounded_statements', []))
        }
        
        # Save AI response with quality metrics
        ai_message = MessageRepository.create(
            db=db,
            conversation_id=conversation_id,
            role="assistant",
            content=result,
            metadata=response_metadata
        )
        db.commit()
        
        # Extract sources with page numbers for provenance
        sources = []
        provenance_dict = {}
        
        for i, doc in enumerate(relevant_docs):
            if hasattr(doc, 'metadata'):
                filename = doc.metadata.get('filename', 'Document')
                page_num = doc.metadata.get('page_number', doc.metadata.get('chunk_index', '?'))
                source_name = f"{filename} (Page {page_num})"
                sources.append(source_name)
                
                # Build provenance dictionary
                provenance_dict[f"source_{i}"] = {
                    "filename": filename,
                    "page": page_num,
                    "content_preview": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                }
            else:
                sources.append("Unknown source")
                provenance_dict[f"source_{i}"] = {
                    "filename": "Unknown",
                    "page": "?",
                    "content_preview": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                }
        
        # Remove duplicates while preserving order
        sources = list(dict.fromkeys(sources))
        
        return ChatResponse(
            response=result,
            conversation_id=conversation_id,
            sources=sources,
            provenance=provenance_dict if provenance_dict else None,
            hallucination_score=hallucination_check['hallucination_score'],
            is_grounded=not hallucination_check['is_hallucinating']
        )

        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"❌ Chat error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Global model cache for whisper
whisper_model = None

def get_whisper_model():
    """Load whisper model once and cache it"""
    global whisper_model
    if whisper_model is None:
        print("Loading Whisper model (this may take a moment)...")
        whisper_model = whisper.load_model("base")
    return whisper_model

@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio file to text using Whisper"""
    if not WHISPER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Whisper is not installed. Run: pip install openai-whisper"
        )
    
    try:
        # Read the audio file content
        content = await file.read()
        print(f"📊 Received audio file: {len(content)} bytes from {file.filename}")
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Get cached model
            model_whisper = get_whisper_model()
            
            # Try to load as WAV first, if that fails, load as raw audio
            audio = None
            sample_rate = 16000
            
            try:
                # Try to open as WAV file
                with wave.open(tmp_path, 'rb') as wav_file:
                    n_channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                    sample_rate = wav_file.getframerate()
                    n_frames = wav_file.getnframes()
                    audio_data = wav_file.readframes(n_frames)
                    
                    print(f"✓ WAV file: {n_channels} ch, {sample_width*8}bit, {sample_rate}Hz, {n_frames} frames ({n_frames/sample_rate:.2f}s)")
                    
                    # Convert bytes to numpy array based on sample width
                    if sample_width == 2:
                        audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    elif sample_width == 1:
                        audio = np.frombuffer(audio_data, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
                    else:
                        raise ValueError(f"Unsupported sample width: {sample_width}")
                    
                    # If stereo, take first channel
                    if n_channels > 1:
                        audio = audio.reshape(-1, n_channels)[:, 0]
                        print(f"  → Converted stereo to mono (using first channel)")
                    
                    print(f"  → Audio shape: {audio.shape}, dtype: {audio.dtype}")
            except Exception as wav_error:
                print(f"⚠️  WAV parsing failed: {str(wav_error)}")
                print("   Attempting to load as raw 16-bit PCM...")
                
                try:
                    # If WAV parsing fails, treat as raw PCM audio (16-bit, 16kHz, mono)
                    audio_data = content
                    audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    sample_rate = 16000
                    print(f"✓ Raw PCM loaded: {len(audio)} samples at {sample_rate}Hz ({len(audio)/sample_rate:.2f}s)")
                except Exception as raw_error:
                    print(f"❌ Raw PCM parsing also failed: {str(raw_error)}")
                    raise
            
            # Check audio levels BEFORE any processing
            if len(audio) > 0:
                max_amplitude = np.max(np.abs(audio))
                mean_amplitude = np.mean(np.abs(audio))
                print(f"📊 Audio levels - Max: {max_amplitude:.6f}, Mean: {mean_amplitude:.6f}")
            else:
                print(f"❌ Audio array is empty!")
                return TranscribeResponse(text="")
            
            # Resample to 16kHz if needed (Whisper expects 16kHz)
            if sample_rate != 16000:
                num_samples = int(len(audio) * 16000 / sample_rate)
                audio = signal.resample(audio, num_samples)
                print(f"📈 Resampled from {sample_rate}Hz to 16000Hz: {len(audio)} samples")
            
            # Transcribe with fp16=False for CPU
            print("🎙️  Sending to Whisper for transcription...")
            result = model_whisper.transcribe(audio, language="en", fp16=False)
            text = result.get("text", "").strip()
            
            print(f"📝 Whisper result: '{text}'")
            
            if text:
                print(f"✅ Successfully transcribed: {text[:100]}{'...' if len(text) > 100 else ''}")
            else:
                print("⚠️  Empty transcription - no speech detected")
            
            return TranscribeResponse(text=text)
            
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"❌ Transcription error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error transcribing audio: {str(e)}")

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    file_type: str = Form(...),
    conversation_id: Optional[str] = Form(None),
    db: Session = Depends(get_db_dep)
):
    """Upload and process PDF or image files"""
    try:
        # Create new conversation if needed
        if not conversation_id:
            conversation = ConversationRepository.create(db)
            db.commit()
            conversation_id = conversation.id
        else:
            # Verify conversation exists
            conversation = ConversationRepository.get_by_id(db, conversation_id)
            if not conversation:
                raise HTTPException(status_code=404, detail="Conversation not found")
            ConversationRepository.update_timestamp(db, conversation_id)
            db.commit()
        
        # Read file content
        content = await file.read()
        
        # Save to temporary file (you might want to store in cloud storage)
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Extract text based on file type
            extracted_text = ""
            if file_type == "pdf":
                extracted_text = extract_text_from_pdf(tmp_path)
            elif file_type in ["image", "jpg", "jpeg", "png"]:
                extracted_text = extract_text_from_image(tmp_path)
            
            # Always use extracted text for indexing
            indexing_text = extracted_text
            
            # For images, determine if we have actual OCR text or just fallback
            has_actual_ocr = False
            if file_type in ["image", "jpg", "jpeg", "png"]:
                # Check if the extracted text contains actual OCR (not just metadata)
                # Fallback contains: "Image:", "File format:", "Dimensions:", "Note:"
                has_actual_ocr = not any(keyword in extracted_text for keyword in ["Note:", "File format:", "Dimensions:"])
            else:
                has_actual_ocr = True  # PDFs are always considered to have content
            
            # Save as message - this creates a DB record of the uploaded file
            file_message = MessageRepository.create(
                db=db,
                conversation_id=conversation_id,
                role="system",
                content=f"[File Uploaded: {file.filename}]\n\n{extracted_text}",
                metadata={
                    "file_type": file_type,
                    "filename": file.filename,
                    "file_size": len(content),
                    "is_uploaded_file": True,
                    "has_ocr_text": has_actual_ocr,
                    "full_extracted_text": extracted_text
                }
            )
            db.commit()

            # Index the document (use extracted text if available, otherwise use fallback)
            # Split by double-newline (page separator) then chunk each page.
            chunks = []
            pages = [p for p in indexing_text.split("\n\n") if p.strip()]
            if not pages:
                pages = [indexing_text]

            # Simple chunking from vector._chunk_text helper via add_documents is fine
            for page in pages:
                # small chunks per page to improve retrieval
                if len(page) <= 1200:
                    chunks.append(page)
                else:
                    # naive fixed-size chunking
                    start = 0
                    while start < len(page):
                        end = min(start + 1000, len(page))
                        chunks.append(page[start:end])
                        if end == len(page):
                            break
                        start = end - 200

            try:
                add_documents(conversation_id, file.filename, chunks)
                print(f"✓ Indexed {file_type} '{file.filename}' with {len(chunks)} chunks")
            except Exception as e:
                print(f"⚠️  Failed to index document: {str(e)}")
                traceback.print_exc()
                # Don't fail upload if indexing fails

            return {
                "success": True,
                "conversation_id": conversation_id,
                "message_id": file_message.id,
                "filename": file.filename
            }
                
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
            
    except Exception as e:
        import traceback
        print(f"❌ Upload error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)