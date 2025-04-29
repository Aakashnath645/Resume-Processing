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
            response = await self.model.generate_content([{"text": prompt}])
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
            experience_score = self._quick_experience_analysis(resume_text, role_level)
            
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
        """Calculate weighted final score based on role level"""
        role_lower = role_level.lower()
        
        if any(term in role_lower for term in ['senior', 'lead', 'manager', 'director']):
            weights = {'keyword': 0.2, 'technical': 0.4, 'experience': 0.4}
        elif any(term in role_lower for term in ['junior', 'intern', 'trainee']):
            weights = {'keyword': 0.4, 'technical': 0.4, 'experience': 0.2}
        else:
            weights = {'keyword': 0.3, 'technical': 0.4, 'experience': 0.3}
        
        final_score = (
            keyword_score * weights['keyword'] +
            technical_score * weights['technical'] +
            experience_score * weights['experience']
        )
        
        return min(100, max(0, final_score))

    def _quick_technical_analysis(self, resume_text: str, job_description: str) -> float:
        """Improved technical skills matching"""
        try:
            # Extract required skills from job description
            required_skills = self._extract_skills(job_description)
            if not required_skills:
                return 50.0  # Default score if no skills found
            
            # Find matching skills in resume
            resume_skills = self._extract_skills(resume_text)
            
            # Calculate exact matches
            exact_matches = len(required_skills.intersection(resume_skills))
            
            # Calculate partial matches
            partial_matches = sum(1 for req in required_skills 
                                for res in resume_skills 
                                if req in res or res in req)
            
            # Calculate scores
            exact_score = (exact_matches / len(required_skills)) * 100
            partial_score = (partial_matches / len(required_skills)) * 20  # Bonus points
            
            return min(100, exact_score + partial_score)
        except:
            return 50.0

    @lru_cache(maxsize=128)
    def _extract_skills(self, text: str) -> set:
        """Cached skill extraction"""
        skills_pattern = r'\b(python|java|javascript|sql|aws|azure|docker|kubernetes|react|angular|node\.js|machine learning|ai|data science|devops)\b'
        return set(re.findall(skills_pattern, text.lower()))

    def _quick_experience_analysis(self, resume_text: str, role_level: str) -> float:
        """Fast experience analysis"""
        try:
            # Extract years of experience
            years_pattern = r'(\d+)[\+]?\s*years?'
            years_matches = re.findall(years_pattern, resume_text.lower())
            years = max([int(y) for y in years_matches]) if years_matches else 0
            
            # Quick scoring based on role level
            if 'senior' in role_level.lower():
                return min(100, years * 10)  # 10 points per year
            elif 'junior' in role_level.lower():
                return min(100, years * 20)  # 20 points per year
            return min(100, years * 15)  # 15 points per year for mid-level
        except:
            return 50.0

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
        """Fast score calculation"""
        try:
            # Weight distribution based on role
            if 'senior' in role_level.lower():
                weights = {'technical': 0.3, 'experience': 0.4, 'keyword': 0.3}
            elif 'junior' in role_level.lower():
                weights = {'technical': 0.4, 'experience': 0.2, 'keyword': 0.4}
            else:
                weights = {'technical': 0.35, 'experience': 0.35, 'keyword': 0.3}
            
            # Calculate overall score
            overall = sum(results[k] * weights[k] for k in weights)
            
            return {
                'overall': min(100, overall),
                'detailed': {
                    'technical': results['technical'],
                    'experience': results['experience'],
                    'keyword': results['keyword']
                }
            }
        except:
            return self._get_default_scores()['scores']

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
