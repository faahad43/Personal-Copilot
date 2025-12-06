from sqlalchemy.orm import Session
from sqlalchemy import desc, asc
from models import Conversation, Message
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

class ConversationRepository:
    @staticmethod
    def create(db: Session, title: str = None) -> Conversation:
        # Generate default title with timestamp
        if title is None:
            title = f"Chat {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
        conversation = Conversation(
            id=str(uuid.uuid4()),
            title=title,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(conversation)
        db.flush()
        return conversation
    
    @staticmethod
    def get_all(db: Session, limit: int = 50) -> List[Conversation]:
        return db.query(Conversation).order_by(desc(Conversation.updated_at)).limit(limit).all()
    
    @staticmethod
    def get_by_id(db: Session, conversation_id: str) -> Optional[Conversation]:
        return db.query(Conversation).filter(Conversation.id == conversation_id).first()
    
    @staticmethod
    def update_title(db: Session, conversation_id: str, title: str) -> Optional[Conversation]:
        conversation = ConversationRepository.get_by_id(db, conversation_id)
        if conversation:
            conversation.title = title
            conversation.updated_at = datetime.utcnow()
        return conversation
    
    @staticmethod
    def delete(db: Session, conversation_id: str) -> bool:
        conversation = ConversationRepository.get_by_id(db, conversation_id)
        if conversation:
            db.delete(conversation)
            return True
        return False
    
    @staticmethod
    def update_timestamp(db: Session, conversation_id: str) -> Optional[Conversation]:
        conversation = ConversationRepository.get_by_id(db, conversation_id)
        if conversation:
            conversation.updated_at = datetime.utcnow()
        return conversation

class MessageRepository:
    @staticmethod
    def create(
        db: Session, 
        conversation_id: str, 
        role: str, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            message_metadata=metadata,
            created_at=datetime.utcnow()
        )
        db.add(message)
        db.flush()
        return message
    
    @staticmethod
    def get_by_conversation(db: Session, conversation_id: str, limit: int = 100) -> List[Message]:
        return db.query(Message)\
            .filter(Message.conversation_id == conversation_id)\
            .order_by(asc(Message.created_at))\
            .limit(limit)\
            .all()
    
    @staticmethod
    def get_last_n_messages(db: Session, conversation_id: str, n: int = 10) -> List[Message]:
        return db.query(Message)\
            .filter(Message.conversation_id == conversation_id)\
            .order_by(desc(Message.created_at))\
            .limit(n)\
            .all()[::-1]  # Reverse to get chronological order
    
    @staticmethod
    def delete_by_conversation(db: Session, conversation_id: str) -> int:
        deleted_count = db.query(Message)\
            .filter(Message.conversation_id == conversation_id)\
            .delete()
        return deleted_count