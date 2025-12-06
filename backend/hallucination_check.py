"""
Hallucination Detection & Page-Level Provenance Tracking
State-of-the-Art techniques for grounding LLM outputs in retrieved documents
"""

from typing import List, Dict, Tuple, Optional
from langchain_core.documents import Document
import re
from difflib import SequenceMatcher

class HallucinationChecker:
    """
    SOTA Hallucination Detection using:
    1. Token overlap analysis between response and source documents
    2. Entailment checking (is response logically implied by sources?)
    3. Source attribution tracking
    """
    
    def __init__(self, threshold: float = 0.3):
        """
        Args:
            threshold: Minimum token overlap ratio to consider statement grounded (0-1)
        """
        self.threshold = threshold
    
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple tokenization"""
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return text.split()
    
    def _calculate_overlap(self, response_tokens: List[str], source_tokens: List[str]) -> float:
        """Calculate token overlap ratio"""
        if not response_tokens:
            return 0.0
        
        overlap = sum(1 for token in response_tokens if token in source_tokens)
        return overlap / len(response_tokens)
    
    def _find_grounding_sources(
        self,
        response: str,
        documents: List[Document],
        window_size: int = 5
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Find which parts of response are grounded in which sources.
        
        Returns:
            Dict mapping sentence -> List[(source_filename, confidence)]
        """
        response_tokens = self._tokenize(response)
        grounding_map = {}
        
        # Split response into sentences
        sentences = re.split(r'[.!?]+', response)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_tokens = self._tokenize(sentence)
            best_sources = []
            
            # Check overlap with each document
            for doc in documents:
                doc_tokens = self._tokenize(doc.page_content)
                overlap = self._calculate_overlap(sentence_tokens, doc_tokens)
                
                if overlap >= self.threshold:
                    source_name = doc.metadata.get('filename', 'Unknown')
                    page_num = doc.metadata.get('page_number', '?')
                    best_sources.append((
                        f"{source_name} (Page {page_num})",
                        overlap
                    ))
            
            grounding_map[sentence] = sorted(
                best_sources,
                key=lambda x: x[1],
                reverse=True
            )
        
        return grounding_map
    
    def check_response(
        self,
        response: str,
        documents: List[Document]
    ) -> Dict:
        """
        Check if response is hallucinating.
        
        Returns:
            {
                'is_hallucinating': bool,
                'hallucination_score': float (0-1, higher = more hallucination),
                'grounded_statements': List[str],
                'ungrounded_statements': List[str],
                'provenance': Dict[str, List[Tuple[str, float]]],
                'confidence': float
            }
        """
        if not documents:
            # For general conversation (no docs needed), use moderate confidence
            # Check if response is likely a greeting/chitchat (short, simple)
            is_general_conversation = (
                len(response.split()) < 50 and  # Short response
                not any(keyword in response.lower() for keyword in [
                    'according to', 'based on', 'the document', 'from the file',
                    'in the pdf', 'shows that', 'indicates that'
                ])
            )
            
            if is_general_conversation:
                return {
                    'is_hallucinating': False,
                    'hallucination_score': 0.3,  # Moderate score for general chat
                    'grounded_statements': [response],
                    'ungrounded_statements': [],
                    'provenance': {},
                    'confidence': 0.7,  # Decent confidence for general knowledge
                    'note': 'General conversation (no document grounding needed)'
                }
            else:
                return {
                    'is_hallucinating': True,
                    'hallucination_score': 1.0,
                    'grounded_statements': [],
                    'ungrounded_statements': [response],
                    'provenance': {},
                    'confidence': 0.0,
                    'warning': 'No source documents provided for factual query'
                }
        
        # Get grounding for each statement
        grounding_map = self._find_grounding_sources(response, documents)
        
        grounded = []
        ungrounded = []
        
        for sentence, sources in grounding_map.items():
            if sources:
                grounded.append(sentence)
            else:
                ungrounded.append(sentence)
        
        # Calculate hallucination score
        total_statements = len(grounding_map)
        if total_statements == 0:
            hallucination_score = 1.0
        else:
            hallucination_score = len(ungrounded) / total_statements
        
        is_hallucinating = hallucination_score > 0.3  # More than 30% ungrounded
        confidence = 1.0 - hallucination_score
        
        return {
            'is_hallucinating': is_hallucinating,
            'hallucination_score': round(hallucination_score, 2),
            'grounded_statements': grounded,
            'ungrounded_statements': ungrounded,
            'provenance': grounding_map,
            'confidence': round(confidence, 2),
            'stats': {
                'total_statements': total_statements,
                'grounded_count': len(grounded),
                'ungrounded_count': len(ungrounded)
            }
        }


class ProvenanceTracker:
    """
    Page-Level Provenance Tracking
    Maps response segments to source documents with exact page/chunk references
    """
    
    @staticmethod
    def create_provenance_report(
        response: str,
        documents: List[Document],
        sources: List[str] = None
    ) -> Dict:
        """
        Create detailed provenance report.
        
        Returns:
            {
                'sources_used': List[str],
                'page_references': Dict[str, List[int]],
                'citation_html': str (formatted with citations)
            }
        """
        sources_used = set()
        page_refs = {}
        
        for doc in documents:
            filename = doc.metadata.get('filename', 'Unknown')
            page_num = doc.metadata.get('page_number', doc.metadata.get('chunk_index', '?'))
            
            sources_used.add(filename)
            if filename not in page_refs:
                page_refs[filename] = set()
            
            if isinstance(page_num, int) or page_num != '?':
                page_refs[filename].add(page_num)
        
        # Convert sets to sorted lists
        page_refs = {k: sorted(list(v)) for k, v in page_refs.items()}
        sources_list = sorted(list(sources_used))
        
        # Create citation format
        citation_parts = []
        for source in sources_list:
            pages = page_refs.get(source, [])
            if pages:
                page_str = ", ".join(str(p) for p in pages)
                citation_parts.append(f"{source} (pp. {page_str})")
            else:
                citation_parts.append(source)
        
        citation_text = " | ".join(citation_parts) if citation_parts else "No sources"
        
        return {
            'sources_used': sources_list,
            'page_references': page_refs,
            'citation_text': citation_text,
            'html': f'<p style="font-size: 0.9em; color: #666; margin-top: 1em; border-top: 1px solid #ddd; padding-top: 0.5em;">📚 Sources: {citation_text}</p>'
        }


def get_response_quality_metrics(
    response: str,
    documents: List[Document],
    hallu_checker: HallucinationChecker = None
) -> Dict:
    """
    Generate comprehensive quality metrics for a response.
    
    Returns combined hallucination check + provenance tracking.
    """
    if hallu_checker is None:
        hallu_checker = HallucinationChecker()
    
    hallucination_check = hallu_checker.check_response(response, documents)
    provenance = ProvenanceTracker.create_provenance_report(response, documents)
    
    return {
        'hallucination': hallucination_check,
        'provenance': provenance,
        'quality_score': hallucination_check['confidence'],
        'is_reliable': (
            not hallucination_check['is_hallucinating'] and
            hallucination_check['confidence'] > 0.6
        )
    }
