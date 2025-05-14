import os
from google import genai
from google.genai import types
from typing import Generator, Dict, Any
import json

class GeminiClient:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.0-flash"

    def generate_stream(self, prompt: str, temperature: float = 0.7) -> Generator[str, None, None]:
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            ),
        ]
        
        config = types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="text/plain",
        )

        try:
            response_stream = self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config,
            )
            
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"Error: {str(e)}"

    def generate_analysis(self, prompt: str) -> Dict[str, Any]:
        """Generate single complete response with error handling"""
        try:
            full_response = ""
            for chunk in self.generate_stream(prompt, temperature=0.3):
                full_response += chunk
            
            return {
                'text': full_response,
                'success': True
            }
        except Exception as e:
            return {
                'text': str(e),
                'success': False
            }
