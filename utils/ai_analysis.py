import os
from dotenv import load_dotenv
import google.generativeai as genai
import logging
import re
import time
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import numpy as np
from functools import lru_cache
import asyncio
from collections import defaultdict
from .gemini_client import GeminiClient

class AIAnalyzer:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        try:
            self.gemini = GeminiClient(api_key)
            self.requests_remaining = 60
            self.last_request_time = 0
            self.min_request_interval = 1.0
        except Exception as e:
            raise ValueError(f"Failed to initialize Gemini API: {str(e)}")
            
        self.max_retries = 2
        self.retry_delay = 0.5
        self.batch_size = 5
        self.cache_size = 1024

    def _wait_for_rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

    def _handle_api_error(self, e: Exception) -> Dict[str, Any]:
        """Handle API errors with appropriate responses"""
        error_msg = str(e).lower()
        
        if "quota" in error_msg or "rate limit" in error_msg:
            # Use fallback analysis for rate limits
            logging.warning("Rate limit reached, using fallback analysis")
            return self._get_fallback_analysis()
        elif "invalid api key" in error_msg:
            logging.error("Invalid API key")
            return self._get_default_scores()
        else:
            logging.error(f"API error: {str(e)}")
            return self._get_default_scores()

    def _get_fallback_analysis(self) -> Dict[str, Any]:
        """Provide basic analysis when API is unavailable"""
        return {
            'scores': {
                'overall': 50,
                'detailed': {
                    'technical': 50,
                    'experience': 50,
                    'keyword': 50
                }
            },
            'analysis_complete': True,
            'using_fallback': True
        }

    @lru_cache(maxsize=1024)
    def get_cached_analysis(self, text_hash: str) -> Dict[str, Any]:
        """Cached analysis results using text hash"""
        return {}

    def batch_process_resumes(self, resumes: List[tuple], job_description: str, role_level: str) -> List[Dict]:
        """Process multiple resumes in parallel batches"""
        results = []
        
        # Create batches
        batches = [resumes[i:i + self.batch_size] for i in range(0, len(resumes), self.batch_size)]
        
        with ProcessPoolExecutor() as executor:
            # Process each batch in parallel
            futures = []
            for batch in batches:
                future = executor.submit(
                    self._process_batch,
                    batch,
                    job_description,
                    role_level
                )
                futures.append(future)
            
            # Collect results
            for future in futures:
                results.extend(future.result())
        
        return results

    def _process_batch(self, batch: List[tuple], job_description: str, role_level: str) -> List[Dict]:
        """Process a batch of resumes"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = []
            for resume_text, _ in batch:
                future = executor.submit(
                    self._analyze_single_resume,
                    resume_text,
                    job_description,
                    role_level
                )
                futures.append(future)
            
            for future in futures:
                try:
                    result = future.result(timeout=30)  # 30 second timeout
                    results.append(result)
                except Exception as e:
                    results.append(self._get_default_scores())
        
        return results

    def _analyze_single_resume(self, resume_text: str, job_description: str, role_level: str) -> Dict[str, Any]:
        """Analyze single resume with unbiased approach"""
        try:
            self._wait_for_rate_limit()
            
            prompt = f"""Objectively analyze this candidate's qualifications for the {role_level} position based solely on skills, experience, and qualifications:

            Required: Evaluate based only on:
            1. Relevant skills and qualifications (0-100)
            2. Applicable experience and achievements (0-100)
            3. Role-specific competencies (0-100)
            4. Overall role alignment (0-100)

            Context:
            Position Requirements: {job_description[:1000]}
            Candidate Background: {resume_text[:1000]}

            Important:
            - Evaluate objectively without considering age, gender, nationality, or background
            - Focus on concrete skills and verifiable experience
            - Consider diverse forms of relevant experience
            - Assess transferable skills from different industries
            - Use consistent criteria across all candidates
            
            Provide clear scoring rationale."""

            response = self.gemini.generate_analysis(prompt)
            
            if not response['success']:
                return self._handle_api_error(Exception(response['text']))
                
            return {
                'scores': self._extract_scores(response['text']),
                'analysis': response['text'],
                'analysis_complete': True
            }
        except Exception as e:
            return self._handle_api_error(e)

    async def get_gemini_response(self, prompt: str) -> Dict[str, Any]:
        """Get response from Gemini API with retries"""
        try:
            response = await self.model.generate_content([{"text": prompt}]) # type: ignore
            return {
                'content': response.text,
                'scores': self._extract_scores(response.text)
            }
        except Exception as e:
            logging.error(f"Gemini API error: {str(e)}")
            return {'content': '', 'scores': {}}

    def parallel_analysis(self, resume_text: str, job_description: str, role_level: str) -> Dict[str, Any]:
        """Main analysis method that combines all scoring components"""
        try:
            # Quick keyword matching first (always works)
            keyword_score = self._calculate_keyword_score(resume_text, job_description)
            
            # Technical skills analysis
            technical_score = self._quick_technical_analysis(resume_text, job_description)
            
            # Experience analysis
            experience_score = self._quick_experience_analysis(resume_text, role_level) # type: ignore
            
            # Calculate final score
            final_score = self._calculate_weighted_score(
                keyword_score=keyword_score,
                technical_score=technical_score,
                experience_score=experience_score,
                role_level=role_level
            )

            return {
                'scores': {
                    'overall': final_score,
                    'detailed': {
                        'technical': technical_score,
                        'experience': experience_score,
                        'keyword': keyword_score
                    }
                },
                'analysis_complete': True
            }
        except Exception as e:
            print(f"Analysis error: {str(e)}")
            return self._get_default_scores()

    def _calculate_weighted_score(self, keyword_score: float, technical_score: float, 
                                experience_score: float, role_level: str) -> float:
        """Calculate weighted score with improved accuracy"""
        role_lower = role_level.lower()
        
        # Define base thresholds
        thresholds = {
            'senior': {'technical': 65, 'experience': 60, 'keyword': 50},
            'junior': {'technical': 40, 'experience': 30, 'keyword': 45},
            'mid': {'technical': 55, 'experience': 45, 'keyword': 45}
        }
        
        # Get appropriate threshold
        if any(term in role_lower for term in ['senior', 'lead', 'manager', 'director']):
            current_threshold = thresholds['senior']
            weights = {'technical': 0.35, 'experience': 0.40, 'keyword': 0.25}
        elif any(term in role_lower for term in ['junior', 'intern', 'trainee']):
            current_threshold = thresholds['junior']
            weights = {'technical': 0.45, 'experience': 0.15, 'keyword': 0.40}
        else:
            current_threshold = thresholds['mid']
            weights = {'technical': 0.40, 'experience': 0.35, 'keyword': 0.25}
        
        # Apply threshold adjustments
        if technical_score < current_threshold['technical']:
            technical_score *= 0.8
        if experience_score < current_threshold['experience']:
            experience_score *= 0.8
        if keyword_score < current_threshold['keyword']:
            keyword_score *= 0.9
        
        # Calculate final score
        final_score = (
            technical_score * weights['technical'] +
            experience_score * weights['experience'] +
            keyword_score * weights['keyword']
        )
        
        # Apply role-specific modifiers
        if 'senior' in role_lower:
            if final_score > 80:
                final_score *= 0.95  # Stricter scoring for senior roles
        elif 'junior' in role_lower:
            if final_score < 45:
                final_score *= 1.15  # More lenient for junior roles
        
        return min(100, max(0, final_score))

    def _quick_technical_analysis(self, resume_text: str, job_description: str) -> float:
        """Improved technical skills matching with relevancy checks"""
        try:
            # Extract skills with importance levels
            required_skills = self._extract_skills_with_importance(job_description)
            if not required_skills:
                return 50.0
            
            resume_skills = self._extract_skills(resume_text) # type: ignore
            
            # Calculate weighted skill matches
            total_score = 0
            total_weight = sum(weight for _, weight in required_skills.items())
            
            for skill, weight in required_skills.items():
                if any(skill in res_skill for res_skill in resume_skills):
                    # Full match
                    if skill in resume_skills:
                        total_score += weight
                    # Partial match
                    else:
                        total_score += weight * 0.5
            
            # Normalize score
            final_score = (total_score / total_weight) * 100 if total_weight > 0 else 50
            
            # Adjust score based on skill context
            skill_context_bonus = self._analyze_skill_context(resume_text, required_skills.keys()) # type: ignore
            
            return min(100, final_score + skill_context_bonus)
        except:
            return 50.0

    def _extract_skills_with_importance(self, text: str) -> dict:
        """Extract skills with importance weights"""
        # Define skills with their importance weights
        skill_weights = {
            'required': 1.0,
            'preferred': 0.7,
            'desired': 0.5,
            'optional': 0.3
        }
        
        skills = {}
        text_lower = text.lower()
        
        # Extract skills and assign weights based on context
        for skill in self._get_common_skills():
            if skill in text_lower:
                context = self._get_skill_context(text_lower, skill)
                skills[skill] = skill_weights.get(context, 0.5)
        
        return skills

    def _analyze_skill_context(self, text: str, skills: set) -> float:
        """Analyze how skills are used in context"""
        context_score = 0
        text_lower = text.lower()
        
        experience_indicators = [
            'developed', 'implemented', 'designed', 'managed',
            'created', 'built', 'maintained', 'led', 'architected'
        ]
        
        for skill in skills:
            # Look for experience indicators near skill mentions
            skill_context = self._get_surrounding_text(text_lower, skill)
            if any(indicator in skill_context for indicator in experience_indicators):
                context_score += 5  # Bonus for demonstrated experience
                
        return min(20, context_score)  # Cap bonus at 20 points

    def _get_surrounding_text(self, text: str, skill: str, window: int = 50) -> str:
        """Get text surrounding a skill mention"""
        try:
            start = max(0, text.index(skill) - window)
            end = min(len(text), text.index(skill) + window)
            return text[start:end]
        except ValueError:
            return ""

    def _get_common_skills(self) -> set:
        """Get common technical skills"""
        return {
            'python', 'java', 'javascript', 'sql', 'aws', 'azure',
            'docker', 'kubernetes', 'react', 'angular', 'node.js',
            'machine learning', 'ai', 'data science', 'devops',
            'cloud', 'agile', 'ci/cd', 'git', 'rest api'
        }

    def _get_skill_context(self, text: str, skill: str) -> str:
        """Determine skill importance from context"""
        if f"required {skill}" in text or f"must have {skill}" in text:
            return "required"
        elif f"preferred {skill}" in text:
            return "preferred"
        elif f"desired {skill}" in text:
            return "desired"
        elif f"optional {skill}" in text:
            return "optional"
        return "normal"

    def _calculate_keyword_score(self, resume_text: str, job_description: str) -> float:
        """Enhanced keyword matching with better accuracy"""
        try:
            # Clean and tokenize texts
            resume_words = set(resume_text.lower().split())
            job_words = set(job_description.lower().split())
            
            # Remove common stop words
            stop_words = {'and', 'or', 'the', 'in', 'at', 'to', 'for', 'a', 'an'}
            resume_words = resume_words - stop_words
            job_words = job_words - stop_words
            
            if not job_words:
                return 50.0
            
            # Calculate matches
            exact_matches = len(resume_words.intersection(job_words))
            partial_matches = sum(1 for job_word in job_words 
                                if any(job_word in resume_word for resume_word in resume_words))
            
            # Calculate scores
            exact_score = (exact_matches / len(job_words)) * 100
            partial_score = (partial_matches / len(job_words)) * 20  # Bonus points
            
            return min(100, exact_score + partial_score)
        except:
            return 50.0

    def _calculate_final_scores(self, results: Dict, role_level: str) -> Dict[str, float]:
        """Fast score calculation with hold recommendation handling"""
        try:
            # Weight distribution based on role
            if 'senior' in role_level.lower():
                weights = {'technical': 0.3, 'experience': 0.4, 'keyword': 0.3}
                hold_threshold = (55, 65)  # Hold range for senior roles
            elif 'junior' in role_level.lower():
                weights = {'technical': 0.4, 'experience': 0.2, 'keyword': 0.4}
                hold_threshold = (35, 50)  # Hold range for junior roles
            else:
                weights = {'technical': 0.35, 'experience': 0.35, 'keyword': 0.3}
                hold_threshold = (45, 60)  # Hold range for mid-level roles
            
            # Calculate overall score
            overall = sum(results[k] * weights[k] for k in weights)
            overall = min(100, overall)
            
            # Determine recommendation
            recommendation = 'hold' if hold_threshold[0] <= overall <= hold_threshold[1] else \
                           'yes' if overall > hold_threshold[1] else 'no'
            
            return {
                'overall': overall,
                'detailed': {
                    'technical': results['technical'],
                    'experience': results['experience'],
                    'keyword': results['keyword']
                },
                'recommendation': recommendation # type: ignore
            }
        except:
            return {
                'overall': 50,
                'detailed': {
                    'technical': 50,
                    'experience': 50,
                    'keyword': 50
                },
                'recommendation': 'hold'  # Default to hold on error # type: ignore
            }

    def _extract_scores(self, text: str) -> Dict[str, float]:
        """Extract numerical scores from Gemini response"""
        scores = {}
        try:
            score_patterns = [
                r'(\w+)\s*:\s*(\d+)',
                r'(\w+)\s*score\s*:\s*(\d+)',
                r'(\w+)\s*match\s*:\s*(\d+)'
            ]
            
            for pattern in score_patterns:
                matches = re.findall(pattern, text.lower())
                for category, score in matches:
                    scores[category.strip()] = min(100, max(0, float(score)))
            
            if not scores:
                scores['overall'] = 50
        except Exception:
            scores['overall'] = 50
            
        return scores

    def _get_default_scores(self) -> Dict[str, Any]:
        """Default scoring structure"""
        return {
            'scores': {
                'overall': 50,
                'detailed': {
                    'technical': 50,
                    'experience': 50,
                    'keyword': 50
                }
            },
            'analysis_complete': False
        }
