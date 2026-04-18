"""Load testing for Ollama Elite AI Platform"""
from locust import HttpUser, task, between
import json

class OllamaUser(HttpUser):
    """Simulates user behavior for load testing"""
    wait_time = between(1, 3)
    
    def on_start(self):
        """Called when a user starts"""
        self.headers = {
            "Authorization": "Bearer dev-key-for-testing-only",
            "Content-Type": "application/json"
        }
    
    @task(2)
    def health_check(self):
        """Health check endpoint (2x frequency)"""
        self.client.get("/health", headers=self.headers)
    
    @task(1)
    def list_models(self):
        """List available models"""
        self.client.get("/api/v1/models", headers=self.headers)
    
    @task(3)
    def generate_text(self):
        """Generate text (3x frequency)"""
        payload = {
            "model": "llama2",
            "prompt": "Hello, how are you?",
            "stream": False
        }
        self.client.post(
            "/api/v1/generate",
            json=payload,
            headers=self.headers
        )

if __name__ == "__main__":
    # Run: locust -f load_test.py --host https://ollama-service-sozvlwbwva-uc.a.run.app
    pass
