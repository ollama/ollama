import requests
from crewai.tools import BaseTool

class SearxNGSearchTool(BaseTool):
    name: str = "SearxNG Search"
    description: str = "A tool to perform searches using a SearxNG instance."
    searxng_instance: str = "https://searx.space/search" # Using a public instance for now

    def _run(self, query: str) -> str:
        try:
            response = requests.get(
                self.searxng_instance,
                params={"q": query, "format": "json"}
            )
            response.raise_for_status()
            results = response.json()
            # Extract and format the results
            snippets = []
            for result in results.get("results", [])[:5]: # Get top 5 results
                snippets.append(f"Title: {result.get('title')}\nURL: {result.get('url')}\nSnippet: {result.get('content')}")
            return "\n\n".join(snippets)
        except requests.exceptions.RequestException as e:
            return f"Error performing search: {e}"

import json
import time
import uuid
import websocket

class ComfyUITool(BaseTool):
    name: str = "ComfyUI Image Generation"
    description: str = "A tool to generate images using a ComfyUI workflow."
    comfyui_host: str = "127.0.0.1:8188"

    def _run(self, prompt: str) -> str:
        client_id = str(uuid.uuid4())
        ws = websocket.WebSocket()
        ws.connect(f"ws://{self.comfyui_host}/ws?clientId={client_id}")

        # A simple workflow to generate an image from a text prompt
        workflow = {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": 156680208700286,
                    "steps": 20,
                    "cfg": 8,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0]
                }
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": "v1-5-pruned-emaonly.ckpt"
                }
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": 512,
                    "height": 512,
                    "batch_size": 1
                }
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
                    "clip": ["4", 1]
                }
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "text, watermark",
                    "clip": ["4", 1]
                }
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2]
                }
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": "ComfyUI",
                    "images": ["8", 0]
                }
            }
        }

        try:
            # Queue the prompt
            data = json.dumps({'prompt': workflow, 'client_id': client_id}).encode('utf-8')
            req = requests.post(f"http://{self.comfyui_host}/prompt", data=data)
            req.raise_for_status()
            prompt_id = req.json()['prompt_id']

            # Wait for the image to be generated
            while True:
                out = ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    if message['type'] == 'executing':
                        data = message['data']
                        if data['node'] is None and data['prompt_id'] == prompt_id:
                            break # Execution is done
                time.sleep(1)

            # Get the history
            history_req = requests.get(f"http://{self.comfyui_host}/history/{prompt_id}")
            history_req.raise_for_status()
            history = history_req.json()
            output_images = history[prompt_id]['outputs']['9']['images']

            # Get the image data
            image_data = output_images[0]
            image_url = f"http://{self.comfyui_host}/view?filename={image_data['filename']}&subfolder={image_data['subfolder']}&type={image_data['type']}"
            return f"Image generated: {image_url}"

        except Exception as e:
            return f"Error generating image: {e}"
        finally:
            ws.close()

if __name__ == '__main__':
    # Example usage
    searx_tool = SearxNGSearchTool()
    search_results = searx_tool.run("latest advancements in AI")
    print("--- SearxNG Search Results ---")
    print(search_results)

    comfy_tool = ComfyUITool()
    # Make sure you have the v1-5-pruned-emaonly.ckpt model downloaded in your ComfyUI/models/checkpoints directory
    image_result = comfy_tool.run("a robot writing a blog post")
    print("\n--- ComfyUI Image Generation Results ---")
    print(image_result)
