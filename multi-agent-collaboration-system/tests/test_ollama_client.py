import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ollama_client import OllamaClient

class TestOllamaClient(unittest.TestCase):

    def setUp(self):
        self.client = OllamaClient()

    def test_initialization(self):
        self.assertEqual(self.client.host, 'http://localhost:11434')

    @patch('requests.post')
    def test_chat_success(self, mock_post):
        # Configure the mock to return a successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'message': {
                'role': 'assistant',
                'content': 'The sky is blue because of Rayleigh scattering.'
            }
        }
        mock_post.return_value = mock_response

        # Call the method being tested
        model = "llama2"
        messages = [{"role": "user", "content": "Why is the sky blue?"}]
        response = self.client.chat(model, messages)

        # Assert the response is as expected
        self.assertEqual(response['message']['content'], 'The sky is blue because of Rayleigh scattering.')
        # Assert that requests.post was called with the correct arguments
        mock_post.assert_called_once_with(
            f"{self.client.host}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False
            }
        )

if __name__ == '__main__':
    unittest.main()
