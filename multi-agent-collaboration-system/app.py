from flask import Flask, render_template, request, jsonify
from ollama_client import OllamaClient
from agents import research_crew, image_crew
import threading

app = Flask(__name__)
ollama_client = OllamaClient()

@app.route('/')
def index():
    # In a real application, you would get the list of models from Ollama
    models = ["llama2", "mistral", "codellama"]
    return render_template('index.html', models=models)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    model = data.get('model', 'llama2')
    messages = data.get('messages', [])

    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    try:
        response = ollama_client.chat(model, messages)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/crew', methods=['POST'])
def crew():
    data = request.get_json()
    crew_type = data.get('crew')
    topic = data.get('topic')

    if not crew_type or not topic:
        return jsonify({"error": "No crew or topic provided"}), 400

    if crew_type == 'research':
        crew_to_run = research_crew
        crew_to_run.tasks[0].description = f"""Conduct a comprehensive analysis of {topic}.
        Identify key trends, breakthrough technologies, and potential industry impacts."""
    elif crew_type == 'image':
        crew_to_run = image_crew
        crew_to_run.tasks[0].description = f"""Generate an image based on the following prompt: {topic}."""
    else:
        return jsonify({"error": "Invalid crew type"}), 400

    # Run the crew in a separate thread to avoid blocking the request
    def run_crew():
        result = crew_to_run.kickoff()
        # In a real application, you would store the result and provide a way to retrieve it.
        print(f"Crew '{crew_type}' finished with result: {result}")

    thread = threading.Thread(target=run_crew)
    thread.start()

    return jsonify({"message": f"Crew '{crew_type}' started. The result will be available in the console."})


if __name__ == '__main__':
    app.run(debug=True, port=5001)
