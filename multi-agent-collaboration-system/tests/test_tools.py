import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools import SearxNGSearchTool, ComfyUITool

class TestTools(unittest.TestCase):

    @patch('requests.get')
    def test_searxng_search_tool(self, mock_get):
        # Configure the mock to return a successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "Test Title",
                    "url": "http://example.com",
                    "content": "Test snippet"
                }
            ]
        }
        mock_get.return_value = mock_response

        # Call the method being tested
        tool = SearxNGSearchTool()
        result = tool.run("test query")

        # Assert the result is as expected
        self.assertIn("Title: Test Title", result)
        self.assertIn("URL: http://example.com", result)
        self.assertIn("Snippet: Test snippet", result)

    @patch('tools.websocket.WebSocket')
    @patch('tools.requests.post')
    @patch('tools.requests.get')
    def test_comfyui_tool(self, mock_get, mock_post, mock_ws):
        # Configure the mock for requests.post
        mock_post_response = MagicMock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = {"prompt_id": "123"}
        mock_post.return_value = mock_post_response

        # Configure the mock for requests.get
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {
            "123": {
                "outputs": {
                    "9": {
                        "images": [
                            {
                                "filename": "test.png",
                                "subfolder": "",
                                "type": "output"
                            }
                        ]
                    }
                }
            }
        }
        mock_get.return_value = mock_get_response

        # Configure the mock for websocket
        mock_ws_instance = mock_ws.return_value
        mock_ws_instance.recv.side_effect = [
            '{"type": "status", "data": {"sid": "123"}}',
            '{"type": "executing", "data": {"node": "some_node", "prompt_id": "123"}}',
            '{"type": "executing", "data": {"node": null, "prompt_id": "123"}}'
        ]


        # Call the method being tested
        tool = ComfyUITool()
        result = tool.run("test prompt")

        # Assert the result is as expected
        self.assertIn("Image generated:", result)
        self.assertIn("test.png", result)


if __name__ == '__main__':
    unittest.main()
