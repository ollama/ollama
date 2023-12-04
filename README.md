# Ollama Web UI: A User-Friendly Web Interface for Chat Interactions 👋

![GitHub stars](https://img.shields.io/github/stars/ollama-webui/ollama-webui?style=social)
![GitHub forks](https://img.shields.io/github/forks/ollama-webui/ollama-webui?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/ollama-webui/ollama-webui?style=social)
![GitHub repo size](https://img.shields.io/github/repo-size/ollama-webui/ollama-webui)
![GitHub language count](https://img.shields.io/github/languages/count/ollama-webui/ollama-webui)
![GitHub top language](https://img.shields.io/github/languages/top/ollama-webui/ollama-webui)
![GitHub last commit](https://img.shields.io/github/last-commit/ollama-webui/ollama-webui?color=red)
![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Follama-webui%2Follama-wbui&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)
[![Discord](https://img.shields.io/badge/Discord-Ollama_Web_UI-blue?logo=discord&logoColor=white)](https://discord.gg/5rJgQTnV4s)
[![](https://img.shields.io/static/v1?label=Sponsor&message=%E2%9D%A4&logo=GitHub&color=%23fe8e86)](https://github.com/sponsors/tjbck)

ChatGPT-Style Web Interface for Ollama 🦙

![Ollama Web UI Demo](./demo.gif)

Also check our sibling project, [OllamaHub](https://ollamahub.com/), where you can discover, download, and explore customized Modelfiles for Ollama! 🦙🔍

## Features ⭐

- 🖥️ **Intuitive Interface**: Our chat interface takes inspiration from ChatGPT, ensuring a user-friendly experience.

- 📱 **Responsive Design**: Enjoy a seamless experience on both desktop and mobile devices.

- ⚡ **Swift Responsiveness**: Enjoy fast and responsive performance.

- 🚀 **Effortless Setup**: Install seamlessly using Docker for a hassle-free experience.

- 💻 **Code Syntax Highlighting**: Enjoy enhanced code readability with our syntax highlighting feature.

- ✒️🔢 **Full Markdown and LaTeX Support**: Elevate your LLM experience with comprehensive Markdown and LaTeX capabilities for enriched interaction.

- 📥🗑️ **Download/Delete Models**: Easily download or remove models directly from the web UI.

- 🤖 **Multiple Model Support**: Seamlessly switch between different chat models for diverse interactions.

- ⚙️ **Many Models Conversations**: Effortlessly engage with various models simultaneously, harnessing their unique strengths for optimal responses. Enhance your experience by leveraging a diverse set of models in parallel.

- 🤝 **OpenAI Model Integration**: Seamlessly utilize OpenAI models alongside Ollama models for a versatile conversational experience.

- 🔄 **Regeneration History Access**: Easily revisit and explore your entire regeneration history.

- 📜 **Chat History**: Effortlessly access and manage your conversation history.

- 📤📥 **Import/Export Chat History**: Seamlessly move your chat data in and out of the platform.

- 🗣️ **Voice Input Support**: Engage with your model through voice interactions; enjoy the convenience of talking to your model directly. Additionally, explore the option for sending voice input automatically after 3 seconds of silence for a streamlined experience.

- ⚙️ **Fine-Tuned Control with Advanced Parameters**: Gain a deeper level of control by adjusting parameters such as temperature and defining your system prompts to tailor the conversation to your specific preferences and needs.

- 🔐 **Auth Header Support**: Effortlessly enhance security by adding Authorization headers to Ollama requests directly from the web UI settings, ensuring access to secured Ollama servers.

- 🔗 **External Ollama Server Connection**: Seamlessly link to an external Ollama server hosted on a different address by configuring the environment variable during the Docker build phase. Additionally, you can also set the external server connection URL from the web UI post-build.

- 🔒 **Backend Reverse Proxy Support**: Strengthen security by enabling direct communication between Ollama Web UI backend and Ollama, eliminating the need to expose Ollama over LAN.

- 🌟 **Continuous Updates**: We are committed to improving Ollama Web UI with regular updates and new features.

## 🔗 Also Check Out OllamaHub! 

Don't forget to explore our sibling project, [OllamaHub](https://ollamahub.com/), where you can discover, download, and explore customized Modelfiles. OllamaHub offers a wide range of exciting possibilities for enhancing your chat interactions with Ollama! 🚀

## How to Install 🚀

### Installing Both Ollama and Ollama Web UI Using Docker Compose

If you don't have Ollama installed yet, you can use the provided Docker Compose file for a hassle-free installation. Simply run the following command:

```bash
docker compose up -d --build
```

This command will install both Ollama and Ollama Web UI on your system. Ensure to modify the `compose.yaml` file for GPU support and Exposing Ollama API outside the container stack if needed.

### Installing Ollama Web UI Only

#### Prerequisites

Make sure you have the latest version of Ollama installed before proceeding with the installation. You can find the latest version of Ollama at [https://ollama.ai/](https://ollama.ai/).

##### Checking Ollama

After installing Ollama, verify that Ollama is running by accessing the following link in your web browser: [http://127.0.0.1:11434/](http://127.0.0.1:11434/). Note that the port number may differ based on your system configuration.

#### Using Docker 🐳

If Ollama is hosted on your local machine and accessible at [http://127.0.0.1:11434/](http://127.0.0.1:11434/), run the following command:

```bash
docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway --name ollama-webui --restart always ghcr.io/ollama-webui/ollama-webui:main
```

Alternatively, if you prefer to build the container yourself, use the following command:

```bash
docker build -t ollama-webui .
docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway --name ollama-webui --restart always ollama-webui
```

Your Ollama Web UI should now be hosted at [http://localhost:3000](http://localhost:3000) and accessible over LAN (or Network). Enjoy! 😄

#### Accessing External Ollama on a Different Server

Change `OLLAMA_API_BASE_URL` environment variable to match the external Ollama Server url:

```bash
docker run -d -p 3000:8080 -e OLLAMA_API_BASE_URL=https://example.com/api --name ollama-webui --restart always ghcr.io/ollama-webui/ollama-webui:main
```

Alternatively, if you prefer to build the container yourself, use the following command:

```bash
docker build -t ollama-webui .
docker run -d -p 3000:8080 -e OLLAMA_API_BASE_URL=https://example.com/api --name ollama-webui --restart always ollama-webui
```

## How to Install Without Docker

While we strongly recommend using our convenient Docker container installation for optimal support, we understand that some situations may require a non-Docker setup, especially for development purposes. Please note that non-Docker installations are not officially supported, and you might need to troubleshoot on your own.

### Project Components

The Ollama Web UI consists of two primary components: the frontend and the backend (which serves as a reverse proxy, handling static frontend files, and additional features). Both need to be running concurrently for the development environment using `npm run dev`. Alternatively, you can set the `PUBLIC_API_BASE_URL` during the build process to have the frontend connect directly to your Ollama instance or build the frontend as static files and serve them with the backend.

### Prerequisites

1. **Clone and Enter the Project:**

   ```sh
   git clone https://github.com/ollama-webui/ollama-webui.git
   cd ollama-webui/
   ```

2. **Create and Edit `.env`:**

   ```sh
   cp -RPp example.env .env
   ```

### Building Ollama Web UI Frontend

1. **Install Node Dependencies:**

   ```sh
   npm install
   ```

2. **Run in Dev Mode or Build for Deployment:**

   - Dev Mode (requires the backend to be running simultaneously):

     ```sh
     npm run dev
     ```

   - Build for Deployment:

     ```sh
     # `PUBLIC_API_BASE_URL` overwrites the value in `.env`
     PUBLIC_API_BASE_URL='https://example.com/api' npm run build
     ```

3. **Test the Build with `Caddy` (or your preferred server):**

   ```sh
   curl https://webi.sh/caddy | sh

   PUBLIC_API_BASE_URL='https://localhost/api' npm run build
   caddy run --envfile .env --config ./Caddyfile.localhost
   ```

### Running Ollama Web UI Backend

If you wish to run the backend for deployment, ensure that the frontend is built so that the backend can serve the frontend files along with the API route.

#### Setup Instructions

1. **Install Python Requirements:**

   ```sh
   cd ./backend
   pip install -r requirements.txt
   ```

2. **Run Python Backend:**

   - Dev Mode with Hot Reloading:

     ```sh
     sh dev.sh
     ```

   - Deployment:

     ```sh
     sh start.sh
     ```

Now, you should have the Ollama Web UI up and running at [http://localhost:8080/](http://localhost:8080/). Feel free to explore the features and functionalities of Ollama! If you encounter any issues, please refer to the instructions above or reach out to the community for assistance.

## Troubleshooting

See [TROUBLESHOOTING.md](/TROUBLESHOOTING.md) for information on how to troubleshoot and/or join our [Ollama Web UI Discord community](https://discord.gg/5rJgQTnV4s).

## What's Next? 🚀

### Roadmap 📝

Here are some exciting tasks on our roadmap:


- 🗃️ **Modelfile Builder**: Easily create Ollama modelfiles via the web UI. Create and add your own character to Ollama by customizing system prompts, conversation starters, and more.
- 🔄 **Multi-Modal Support**: Seamlessly engage with models that support multimodal interactions, including images (e.g., LLava).
- 📚 **RAG Integration**: Experience first-class retrieval augmented generation support, enabling chat with your documents.
- 🔐 **Access Control**: Securely manage requests to Ollama by utilizing the backend as a reverse proxy gateway, ensuring only authenticated users can send specific requests.
- 🧪 **Research-Centric Features**: Empower researchers in the fields of LLM and HCI with a comprehensive web UI for conducting user studies. Stay tuned for ongoing feature enhancements (e.g., surveys, analytics, and participant tracking) to facilitate their research.
- 📈 **User Study Tools**: Providing specialized tools, like heat maps and behavior tracking modules, to empower researchers in capturing and analyzing user behavior patterns with precision and accuracy.
- 📚 **Enhanced Documentation**: Elevate your setup and customization experience with improved, comprehensive documentation.

Feel free to contribute and help us make Ollama Web UI even better! 🙌

## Supporters ✨

A big shoutout to our amazing supporters who's helping to make this project possible! 🙏

### Platinum Sponsors 🤍

- [Prof. Lawrence Kim @ SFU](https://www.lhkim.com/)

## License 📜

This project is licensed under the [MIT License](LICENSE) - see the [LICENSE](LICENSE) file for details. 📄

## Support 💬

If you have any questions, suggestions, or need assistance, please open an issue or join our
[Ollama Web UI Discord community](https://discord.gg/5rJgQTnV4s) or
[Ollama Discord community](https://discord.gg/ollama) to connect with us! 🤝

---

Created by [Timothy J. Baek](https://github.com/tjbck) - Let's make Ollama Web UI even more amazing together! 💪
