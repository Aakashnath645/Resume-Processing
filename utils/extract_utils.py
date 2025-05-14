import re
from typing import Optional

def extract_candidate_name(resume_text: str) -> str:
    """Enhanced candidate name extraction with multiple strategies"""
    if not resume_text:
        return "Unknown_Candidate"
        
    # Clean text and get first few lines
    lines = [line.strip() for line in resume_text.split('\n')[:6] if line.strip()]
    
    # Common resume headers to ignore
    headers = {'resume', 'curriculum vitae', 'cv', 'biodata', 'personal details'}
    
    # Name patterns
    name_patterns = [
        # Standard name format: 2-3 words with optional middle initial
        r'^([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)$',
        
        # Name after labels like "Name:"
        r'(?:name|full name|candidate)[:\s]+([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)',
        
        # Professional prefix with name
        r'(?:mr\.|ms\.|mrs\.|dr\.)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)'
    ]
    
    for line in lines:
        line = line.strip().lower()
        
        # Skip if line is a common header
        if line in headers:
            continue
            
        # Try each pattern
        for pattern in name_patterns:
            matches = re.findall(pattern, line, re.IGNORECASE)
            if matches:
                name = matches[0].strip()
                # Validate name format
                if 2 <= len(name.split()) <= 3:
                    if all(word[0].isupper() for word in name.split()):
                        return name.title()
    
    # Fallback: Look for first capitalized sequence
    for line in lines:
        words = line.split()
        if 2 <= len(words) <= 3:
            if all(word[0].isupper() for word in words):
                if all(word.replace('.','').isalpha() for word in words):
                    return ' '.join(words)
    
    return "Unknown_Candidate"

def validate_name(name: str) -> bool:
    """Validate extracted name"""
    if not name or len(name.split()) < 2:
        return False
        
    words = name.split()
    return all(
        word[0].isupper() and 
        word[1:].islower() and 
        word.isalpha() 
        for word in words
    )
