import re
from typing import Dict, Any

def calculate_role_weights(role_level: str) -> Dict[str, float]:
    """Calculate importance weights based on role level"""
    weights = {
        'technical': 0.3,
        'experience': 0.2,
        'keyword': 0.2,
        'leadership': 0.15,
        'cultural': 0.15
    }
    
    if 'senior' in role_level.lower() or 'lead' in role_level.lower():
        weights.update({
            'technical': 0.25,
            'leadership': 0.25,
            'experience': 0.25
        })
    elif 'junior' in role_level.lower():
        weights.update({
            'technical': 0.35,
            'experience': 0.15,
            'potential': 0.25
        })
    
    return weights

def compute_basic_keyword_score(resume_text: str, job_description: str) -> float:
    """Compute basic keyword matching score"""
    job_words = set(job_description.lower().split())
    resume_words = set(resume_text.lower().split())
    
    common_words = job_words.intersection(resume_words)
    if not job_words:
        return 0
        
    return min(100, (len(common_words) / len(job_words)) * 100)

def is_suitable_numeric(score: float, role_level: str, threshold: float = 45) -> str:
    """Determine if the candidate is suitable based on score and role"""
    role_lower = role_level.lower()
    
    if any(term in role_lower for term in ['senior', 'lead', 'manager']):
        return "Yes" if score >= 40 else "No"
    elif any(term in role_lower for term in ['junior', 'intern']):
        return "Yes" if score >= 35 else "No"
    
    return "Yes" if score >= threshold else "No"
