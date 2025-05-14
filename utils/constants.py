ROLE_WEIGHTS = {
    'Chief': {
        'skills': 0.15,
        'experience': 0.25,
        'leadership': 0.30,
        'project': 0.15,
        'cultural': 0.15
    },
    'Director': {
        'skills': 0.20,
        'experience': 0.25,
        'leadership': 0.25,
        'project': 0.15,
        'cultural': 0.15
    },
    'Senior': {
        'skills': 0.25,
        'experience': 0.20,
        'leadership': 0.20,
        'project': 0.20,
        'cultural': 0.15
    },
    'Manager': {
        'skills': 0.25,
        'experience': 0.20,
        'leadership': 0.20,
        'project': 0.20,
        'cultural': 0.15
    },
    'Associate': {
        'skills': 0.30,
        'experience': 0.15,
        'leadership': 0.15,
        'project': 0.20,
        'cultural': 0.20
    }
}

SKILL_PATTERNS = {
    'programming': r'\b(python|java|javascript|c\+\+|ruby|php)\b',
    'tools': r'\b(git|docker|kubernetes|jenkins|aws|azure)\b',
    'soft_skills': r'\b(leadership|communication|teamwork|management)\b',
    'frameworks': r'\b(react|angular|vue|django|flask|spring)\b',
    'databases': r'\b(sql|mysql|postgresql|mongodb|oracle)\b'
}

EXPERIENCE_INDICATORS = [
    'years', 'year', 'experience', 'expert', 'senior', 'lead',
    'managed', 'developed', 'implemented', 'architected'
]

MIN_EXPERIENCE_YEARS = {
    'Chief': 15,
    'Director': 12,
    'Senior': 8,
    'Manager': 5,
    'Associate': 0
}
