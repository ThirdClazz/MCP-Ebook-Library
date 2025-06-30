#!/usr/bin/env python3

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import time
import pickle
import hashlib

# FastMCP imports
from mcp.server.fastmcp import FastMCP

# For PDF parsing - using multiple libraries for better extraction
import PyPDF2
try:
    import pdfplumber  # Better text extraction
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    print("Note: Install pdfplumber for better PDF text extraction: pip install pdfplumber")

# Initialize FastMCP server
mcp = FastMCP("pdf-textbook-server")

# Configuration
EBOOKS_DIR = Path(os.getenv('EBOOKS_DIR', './ebooks'))
CACHE_DIR = EBOOKS_DIR.parent / "pdf_content_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Content index for fast searching
_content_index = {}
_index_loaded = False

def get_file_hash(file_path: Path) -> str:
    """Get file hash to detect changes"""
    return hashlib.md5(f"{file_path.stat().st_mtime}_{file_path.stat().st_size}".encode()).hexdigest()

def get_bundle_name(file_path: Path) -> str:
    """Extract bundle name from file path"""
    try:
        relative_path = file_path.relative_to(EBOOKS_DIR)
        return str(relative_path.parent) if relative_path.parent != Path('.') else "Root"
    except:
        return "Unknown"

def categorize_book(filename: str) -> str:
    """Categorize books by content type"""
    filename_lower = filename.lower()
    
    if any(term in filename_lower for term in ['python', 'django', 'fastapi']):
        return "Python Programming"
    elif any(term in filename_lower for term in ['machine learning', 'ai', 'neural', 'deep learning', 'gpt', 'llm']):
        return "Machine Learning & AI"
    elif any(term in filename_lower for term in ['visualization', 'tableau', 'power bi', 'excel']):
        return "Data Visualization"
    elif any(term in filename_lower for term in ['dummies', 'math', 'calculus', 'algebra', 'physics']):
        return "Math & Physics"
    elif any(term in filename_lower for term in ['sql', 'database']):
        return "Data Engineering & Databases"
    else:
        return "Technical"

@mcp.tool()
def extract_content_about_topic(topic: str, max_results: int = 10) -> str:
    """
    Extract actual content about a specific topic from all textbooks.
    
    Args:
        topic: Topic to search for (e.g., 'EDA', 'neural networks', 'pandas', 'regression')
        max_results: Maximum number of content chunks to return
    
    Returns:
        JSON with actual textbook content explaining the topic
    """
    load_or_build_content_index()
    
    if not _content_index:
        return json.dumps({"error": "No content index available. Try running build_content_index first."})
    
    topic_lower = topic.lower()
    results = []
    
    # Search through all indexed content
    for file_path, book_data in _content_index.items():
        if 'error' in book_data:
            continue
            
        filename = book_data['filename']
        category = book_data.get('category', 'Unknown')
        bundle = book_data.get('bundle', 'Unknown')
        
        # Search through content sections
        for section in book_data.get('sections', []):
            content = section['content']
            content_lower = content.lower()
            
            # Check if topic appears in this section
            if topic_lower in content_lower:
                # Calculate relevance score
                topic_count = content_lower.count(topic_lower)
                
                # Extract context around the topic
                context = extract_topic_context(content, topic, max_chars=1500)
                
                if context.strip():
                    results.append({
                        'filename': filename,
                        'bundle': bundle,
                        'category': category,
                        'section_title': section.get('title', f"Section {section['section_id']}"),
                        'page_range': section.get('page_range', 'Unknown'),
                        'topic_mentions': topic_count,
                        'relevance_score': calculate_relevance_score(content, topic),
                        'content_excerpt': context,
                        'content_length': len(content)
                    })
    
    # Sort by relevance score
    results.sort(key=lambda x: (-x['relevance_score'], -x['topic_mentions']))
    
    # Limit results
    results = results[:max_results]
    
    return json.dumps({
        'topic': topic,
        'total_matches': len(results),
        'books_searched': len([book for book in _content_index.values() if 'sections' in book]),
        'content_extracts': results
    }, indent=2)

@mcp.tool()
def search_for_code_examples(language_or_topic: str, max_results: int = 8) -> str:
    """
    Find code examples related to a specific programming language or topic.
    
    Args:
        language_or_topic: Programming language or coding topic (e.g., 'python', 'pandas', 'matplotlib', 'sql')
        max_results: Maximum number of code examples to return
    
    Returns:
        JSON with code examples and explanations from textbooks
    """
    load_or_build_content_index()
    
    if not _content_index:
        return json.dumps({"error": "No content index available."})
    
    search_term = language_or_topic.lower()
    code_results = []
    
    # Search for code patterns
    for file_path, book_data in _content_index.items():
        if 'error' in book_data:
            continue
            
        filename = book_data['filename']
        category = book_data.get('category', 'Unknown')
        
        for section in book_data.get('sections', []):
            content = section['content']
            
            # Look for code blocks and relevant content
            if has_code_patterns(content) and search_term in content.lower():
                code_blocks = extract_code_blocks(content)
                
                if code_blocks:
                    for i, code_block in enumerate(code_blocks[:2]):  # Max 2 per section
                        # Get explanation around the code
                        explanation = extract_code_explanation(content, code_block)
                        
                        code_results.append({
                            'filename': filename,
                            'category': category,
                            'section_title': section.get('title', f"Section {section['section_id']}"),
                            'code_block': code_block,
                            'explanation': explanation,
                            'relevance_score': calculate_code_relevance(code_block, explanation, search_term)
                        })
    
    # Sort by relevance
    code_results.sort(key=lambda x: -x['relevance_score'])
    code_results = code_results[:max_results]
    
    return json.dumps({
        'search_term': language_or_topic,
        'total_code_examples': len(code_results),
        'code_examples': code_results
    }, indent=2)

@mcp.tool()
def explain_concept_from_books(concept: str, preferred_category: str = "") -> str:
    """
    Get comprehensive explanations of a concept from your textbooks.
    
    Args:
        concept: Concept to explain (e.g., 'machine learning', 'data normalization', 'neural networks')
        preferred_category: Preferred book category to search in (optional)
    
    Returns:
        JSON with detailed explanations from multiple textbooks
    """
    load_or_build_content_index()
    
    if not _content_index:
        return json.dumps({"error": "No content index available."})
    
    concept_lower = concept.lower()
    explanations = []
    
    for file_path, book_data in _content_index.items():
        if 'error' in book_data:
            continue
            
        # Filter by category if specified
        if preferred_category and preferred_category.lower() not in book_data.get('category', '').lower():
            continue
            
        filename = book_data['filename']
        category = book_data.get('category', 'Unknown')
        
        for section in book_data.get('sections', []):
            content = section['content']
            content_lower = content.lower()
            
            # Look for definitions and explanations
            if concept_lower in content_lower:
                explanation = extract_concept_explanation(content, concept)
                
                if explanation and len(explanation.strip()) > 100:  # Substantial explanation
                    explanations.append({
                        'filename': filename,
                        'category': category,
                        'section_title': section.get('title', f"Section {section['section_id']}"),
                        'explanation': explanation,
                        'explanation_quality': assess_explanation_quality(explanation, concept),
                        'page_info': section.get('page_range', 'Unknown')
                    })
    
    # Sort by explanation quality
    explanations.sort(key=lambda x: -x['explanation_quality'])
    explanations = explanations[:8]  # Top 8 explanations
    
    return json.dumps({
        'concept': concept,
        'category_filter': preferred_category or "All categories",
        'total_explanations': len(explanations),
        'explanations': explanations
    }, indent=2)

@mcp.tool()
def build_content_index() -> str:
    """
    Build or rebuild the content index for all PDF textbooks.
    This processes all PDFs and creates a searchable index. Run this once or when you add new books.
    
    Returns:
        JSON with indexing progress and results
    """
    global _content_index, _index_loaded
    
    print("Building content index for PDF textbooks...")
    start_time = time.time()
    
    # Find all PDF files
    pdf_files = list(EBOOKS_DIR.glob("**/*.pdf"))
    total_files = len(pdf_files)
    
    if total_files == 0:
        return json.dumps({"error": f"No PDF files found in {EBOOKS_DIR}"})
    
    _content_index = {}
    processed = 0
    errors = 0
    
    for i, file_path in enumerate(pdf_files):
        try:
            print(f"Processing {i+1}/{total_files}: {file_path.name}")
            
            # Extract content sections
            sections = extract_pdf_sections(file_path)
            
            _content_index[str(file_path)] = {
                'filename': file_path.name,
                'file_hash': get_file_hash(file_path),
                'bundle': get_bundle_name(file_path),
                'category': categorize_book(file_path.name),
                'sections': sections,
                'total_sections': len(sections),
                'indexed_at': datetime.now().isoformat(),
                'file_size_mb': round(file_path.stat().st_size / (1024 * 1024), 2)
            }
            processed += 1
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            _content_index[str(file_path)] = {
                'filename': file_path.name,
                'error': str(e),
                'indexed_at': datetime.now().isoformat()
            }
            errors += 1
    
    # Save index
    index_file = CACHE_DIR / "content_index.pkl"
    try:
        with open(index_file, 'wb') as f:
            pickle.dump(_content_index, f)
        saved = True
    except Exception as e:
        print(f"Failed to save index: {e}")
        saved = False
    
    _index_loaded = True
    processing_time = round(time.time() - start_time, 2)
    
    return json.dumps({
        'total_files': total_files,
        'processed_successfully': processed,
        'errors': errors,
        'processing_time_seconds': processing_time,
        'index_saved': saved,
        'cache_location': str(index_file),
        'total_sections_indexed': sum(len(book.get('sections', [])) for book in _content_index.values() if 'sections' in book)
    }, indent=2)

@mcp.tool()
def get_index_status() -> str:
    """
    Check the status of the content index.
    
    Returns:
        JSON with current index status and statistics
    """
    global _index_loaded, _content_index
    
    index_file = CACHE_DIR / "content_index.pkl"
    
    if not _index_loaded:
        if index_file.exists():
            try:
                with open(index_file, 'rb') as f:
                    _content_index = pickle.load(f)
                _index_loaded = True
            except:
                pass
    
    status = {
        'index_loaded': _index_loaded,
        'index_file_exists': index_file.exists(),
        'cache_directory': str(CACHE_DIR),
        'textbooks_directory': str(EBOOKS_DIR)
    }
    
    if _index_loaded and _content_index:
        successful_books = [book for book in _content_index.values() if 'sections' in book]
        error_books = [book for book in _content_index.values() if 'error' in book]
        
        status.update({
            'total_books_in_index': len(_content_index),
            'successfully_indexed': len(successful_books),
            'failed_to_index': len(error_books),
            'total_content_sections': sum(len(book.get('sections', [])) for book in successful_books),
            'categories': list(set(book.get('category') for book in successful_books if book.get('category'))),
            'last_indexed': max([book.get('indexed_at', '') for book in _content_index.values()], default='Never')
        })
        
        if error_books:
            status['error_files'] = [book['filename'] for book in error_books]
    
    return json.dumps(status, indent=2)

def load_or_build_content_index():
    """Load existing content index or prompt user to build it"""
    global _content_index, _index_loaded
    
    if _index_loaded:
        return
    
    index_file = CACHE_DIR / "content_index.pkl"
    
    if index_file.exists():
        try:
            with open(index_file, 'rb') as f:
                _content_index = pickle.load(f)
            _index_loaded = True
            print(f"Loaded content index with {len(_content_index)} books")
            return
        except Exception as e:
            print(f"Failed to load existing index: {e}")
    
    # Index doesn't exist - user needs to build it
    _content_index = {}
    _index_loaded = True

def extract_pdf_sections(file_path: Path) -> List[Dict]:
    """Extract content sections from PDF with better text extraction"""
    sections = []
    
    try:
        if HAS_PDFPLUMBER:
            # Use pdfplumber for better text extraction
            import pdfplumber
            
            with pdfplumber.open(file_path) as pdf:
                current_section = ""
                section_id = 0
                page_start = 1
                
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            current_section += page_text + "\n"
                            
                            # Create sections every 10 pages or at chapter boundaries
                            if page_num % 10 == 0 or detect_chapter_boundary(page_text):
                                if current_section.strip():
                                    sections.append({
                                        'section_id': section_id,
                                        'content': current_section.strip(),
                                        'page_range': f"{page_start}-{page_num}",
                                        'title': extract_section_title(current_section)
                                    })
                                    section_id += 1
                                    current_section = ""
                                    page_start = page_num + 1
                                    
                    except Exception as e:
                        continue
                
                # Add final section
                if current_section.strip():
                    sections.append({
                        'section_id': section_id,
                        'content': current_section.strip(),
                        'page_range': f"{page_start}-{len(pdf.pages)}",
                        'title': extract_section_title(current_section)
                    })
        
        else:
            # Fallback to PyPDF2
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                current_section = ""
                section_id = 0
                page_start = 1
                
                for page_num in range(len(pdf_reader.pages)):
                    try:
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        if page_text:
                            current_section += page_text + "\n"
                            
                            # Create sections every 10 pages
                            if (page_num + 1) % 10 == 0:
                                if current_section.strip():
                                    sections.append({
                                        'section_id': section_id,
                                        'content': current_section.strip(),
                                        'page_range': f"{page_start}-{page_num + 1}",
                                        'title': extract_section_title(current_section)
                                    })
                                    section_id += 1
                                    current_section = ""
                                    page_start = page_num + 2
                    except:
                        continue
                
                # Add final section
                if current_section.strip():
                    sections.append({
                        'section_id': section_id,
                        'content': current_section.strip(),
                        'page_range': f"{page_start}-{len(pdf_reader.pages)}",
                        'title': extract_section_title(current_section)
                    })
    
    except Exception as e:
        sections = [{'section_id': 0, 'content': f"Error extracting PDF: {str(e)}", 'page_range': '1-1', 'title': 'Error'}]
    
    return sections

def detect_chapter_boundary(text: str) -> bool:
    """Detect if this page likely contains a chapter boundary"""
    lines = text.split('\n')[:10]  # Check first 10 lines
    for line in lines:
        line = line.strip()
        if re.match(r'^(Chapter|CHAPTER)\s+\d+', line) or \
           re.match(r'^\d+\.\s+[A-Z]', line) or \
           line.isupper() and len(line) > 10 and len(line) < 100:
            return True
    return False

def extract_section_title(content: str) -> str:
    """Extract a title for the section from its content"""
    lines = content.strip().split('\n')[:20]  # Check first 20 lines
    
    for line in lines:
        line = line.strip()
        if line and len(line) < 100:
            # Look for chapter titles, headers, etc.
            if re.match(r'^(Chapter|CHAPTER)\s+\d+', line) or \
               re.match(r'^\d+\.\s+', line) or \
               (line.isupper() and len(line) > 5):
                return line
    
    # Fallback: use first substantial line
    for line in lines:
        line = line.strip()
        if len(line) > 10 and len(line) < 100:
            return line[:80] + "..." if len(line) > 80 else line
    
    return "Content Section"

def extract_topic_context(content: str, topic: str, max_chars: int = 1500) -> str:
    """Extract relevant context around a topic mention"""
    content_lower = content.lower()
    topic_lower = topic.lower()
    
    # Find all occurrences of the topic
    occurrences = []
    start = 0
    while True:
        pos = content_lower.find(topic_lower, start)
        if pos == -1:
            break
        occurrences.append(pos)
        start = pos + 1
    
    if not occurrences:
        return ""
    
    # Get context around the best occurrence (usually the first substantial one)
    best_pos = occurrences[0]
    
    # Expand context to sentence boundaries
    start_pos = max(0, best_pos - max_chars // 2)
    end_pos = min(len(content), best_pos + max_chars // 2)
    
    # Adjust to sentence boundaries
    while start_pos > 0 and content[start_pos] not in '.!?':
        start_pos -= 1
    while end_pos < len(content) and content[end_pos] not in '.!?':
        end_pos += 1
    
    context = content[start_pos:end_pos].strip()
    
    # Clean up the context
    if context.startswith('.') or context.startswith('!') or context.startswith('?'):
        context = context[1:].strip()
    
    return context

def calculate_relevance_score(content: str, topic: str) -> float:
    """Calculate how relevant a content section is to a topic"""
    content_lower = content.lower()
    topic_lower = topic.lower()
    
    # Count topic mentions
    topic_count = content_lower.count(topic_lower)
    
    # Check for definition patterns
    definition_patterns = [
        f"{topic_lower} is",
        f"{topic_lower} refers to",
        f"{topic_lower} means",
        f"what is {topic_lower}",
        f"definition of {topic_lower}"
    ]
    
    definition_score = sum(1 for pattern in definition_patterns if pattern in content_lower)
    
    # Check for related keywords
    related_keywords_score = 0
    if 'example' in content_lower and topic_lower in content_lower:
        related_keywords_score += 1
    if 'implementation' in content_lower and topic_lower in content_lower:
        related_keywords_score += 1
    
    # Calculate final score
    score = topic_count + (definition_score * 2) + related_keywords_score
    
    # Normalize by content length
    return score / max(1, len(content) / 1000)

def has_code_patterns(content: str) -> bool:
    """Check if content contains code patterns"""
    code_indicators = [
        'import ', 'def ', 'class ', 'print(', 'return ',
        '```', 'code:', 'example:', '>>>', 'import pandas',
        'import numpy', 'import matplotlib', 'from sklearn'
    ]
    
    content_lower = content.lower()
    return any(indicator in content_lower for indicator in code_indicators)

def extract_code_blocks(content: str) -> List[str]:
    """Extract code blocks from content"""
    code_blocks = []
    
    # Look for various code block patterns
    patterns = [
        r'```[\w]*\n(.*?)\n```',  # Markdown code blocks
        r'>>> (.*?)(?=\n\n|\n[A-Z]|\Z)',  # Python REPL
        r'((?:import|from|def|class|print|return).*?)(?=\n\n|\n[A-Z]|\Z)'  # Python code patterns
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0] if match else ""
            
            if len(match.strip()) > 20:  # Substantial code block
                code_blocks.append(match.strip())
    
    return code_blocks[:5]  # Limit to 5 code blocks per section

def extract_code_explanation(content: str, code_block: str) -> str:
    """Extract explanation text around a code block"""
    # Find the code block position in content
    code_pos = content.find(code_block)
    if code_pos == -1:
        return "No explanation found"
    
    # Get text before and after the code block
    before_text = content[:code_pos]
    after_text = content[code_pos + len(code_block):]
    
    # Extract explanation (prefer text before the code)
    explanation_lines = []
    
    # Get lines before code
    before_lines = before_text.split('\n')[-5:]  # Last 5 lines before code
    explanation_lines.extend([line.strip() for line in before_lines if line.strip()])
    
    # Get lines after code if not enough before
    if len(' '.join(explanation_lines)) < 100:
        after_lines = after_text.split('\n')[:3]  # First 3 lines after code
        explanation_lines.extend([line.strip() for line in after_lines if line.strip()])
    
    explanation = ' '.join(explanation_lines)
    return explanation[:500] + "..." if len(explanation) > 500 else explanation

def calculate_code_relevance(code_block: str, explanation: str, search_term: str) -> float:
    """Calculate relevance of a code example to the search term"""
    code_lower = code_block.lower()
    explanation_lower = explanation.lower()
    search_lower = search_term.lower()
    
    score = 0
    
    # Search term in code
    score += code_lower.count(search_lower) * 3
    
    # Search term in explanation
    score += explanation_lower.count(search_lower) * 2
    
    # Code quality indicators
    if 'import' in code_lower:
        score += 1
    if 'def ' in code_lower:
        score += 1
    if len(code_block) > 50:  # Substantial code
        score += 1
    
    return score

def extract_concept_explanation(content: str, concept: str) -> str:
    """Extract explanation of a concept from content"""
    concept_lower = concept.lower()
    content_lower = content.lower()
    
    # Find concept mentions
    sentences = re.split(r'[.!?]+', content)
    relevant_sentences = []
    
    for i, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()
        if concept_lower in sentence_lower:
            # Include surrounding sentences for context
            start_idx = max(0, i - 1)
            end_idx = min(len(sentences), i + 3)
            context_sentences = sentences[start_idx:end_idx]
            relevant_sentences.extend(context_sentences)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_sentences = []
    for sentence in relevant_sentences:
        sentence = sentence.strip()
        if sentence and sentence not in seen:
            seen.add(sentence)
            unique_sentences.append(sentence)
    
    explanation = '. '.join(unique_sentences)
    
    # Limit length
    if len(explanation) > 2000:
        explanation = explanation[:2000] + "..."
    
    return explanation

def assess_explanation_quality(explanation: str, concept: str) -> float:
    """Assess the quality of an explanation"""
    explanation_lower = explanation.lower()
    concept_lower = concept.lower()
    
    score = 0
    
    # Length bonus (but not too long)
    length = len(explanation)
    if 100 <= length <= 1000:
        score += 2
    elif length > 1000:
        score += 1
    
    # Definition indicators
    definition_words = ['is', 'refers to', 'means', 'defined as', 'definition']
    for word in definition_words:
        if f"{concept_lower} {word}" in explanation_lower:
            score += 3
            break
    
    # Example indicators
    if 'example' in explanation_lower and concept_lower in explanation_lower:
        score += 2
    
    # Technical terms (good for textbook explanations)
    technical_indicators = ['algorithm', 'method', 'process', 'technique', 'approach']
    score += sum(1 for term in technical_indicators if term in explanation_lower)
    
    return score

if __name__ == '__main__':
    # Create cache directory
    CACHE_DIR.mkdir(exist_ok=True)
    
    print(f"Starting PDF textbook content server")
    print(f"Textbooks directory: {EBOOKS_DIR}")
    print(f"Cache directory: {CACHE_DIR}")
    print(f"PDFPlumber available: {HAS_PDFPLUMBER}")
    
    if not HAS_PDFPLUMBER:
        print("Recommendation: pip install pdfplumber for better PDF text extraction")
    
    # Run the server
    mcp.run()