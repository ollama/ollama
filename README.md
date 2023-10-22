# Ollama Web UI 👋

ChatGPT-Style Web Interface for Ollama 🦙

![Ollama Web UI Demo](./demo.gif)

## Features ⭐

- 🖥️ **Intuitive Interface**: Our chat interface takes inspiration from ChatGPT, ensuring a user-friendly experience.

- 📱 **Responsive Design**: Enjoy a seamless experience on both desktop and mobile devices.

- ⚡ **Swift Responsiveness**: Enjoy fast and responsive performance.

- 🚀 **Effortless Setup**: Install seamlessly using Docker for a hassle-free experience.

- 🤖 **Multiple Model Support**: Seamlessly switch between different chat models for diverse interactions.

- 📜 **Chat History**: Effortlessly access and manage your conversation history.

- 📤📥 **Import/Export Chat History**: Seamlessly move your chat data in and out of the platform.

- ⚙️ **Fine-Tuned Control with Advanced Parameters**: Gain a deeper level of control by adjusting parameters such as temperature and defining your system prompts to tailor the conversation to your specific preferences and needs.

- 💻 **Code Syntax Highlighting**: Enjoy enhanced code readability with our syntax highlighting feature.

- 🔗 **External Ollama Server Connection**: You can seamlessly connect to an external Ollama server hosted on a different address by setting the environment variable during the Docker build process. Execute the following command to include the Ollama API base URL in the Docker image: `docker build --build-arg OLLAMA_API_BASE_URL='http://localhost:11434/api' -t ollama-webui .`.

- 🌟 **Continuous Updates**: We are committed to improving Ollama Web UI with regular updates and new features.

## How to Install 🚀

### Prerequisites

Make sure you have the latest version of Ollama installed before proceeding with the installation. You can find the latest version of Ollama at [https://ollama.ai/](https://ollama.ai/).

#### Checking Ollama

After installing, verify that Ollama is running by accessing the following link in your web browser: [http://127.0.0.1:11434/](http://127.0.0.1:11434/). Note that the port number may differ based on your system configuration.

#### Accessing Ollama Web Interface over LAN

If you want to access the Ollama web interface over LAN, for example, from your phone, run Ollama using the following command:

```bash
OLLAMA_HOST=0.0.0.0 OLLAMA_ORIGINS=* ollama serve
```

### Using Docker 🐳

If Ollama is hosted on your local machine, run the following command:

```bash
docker build --build-arg OLLAMA_API_BASE_URL='' -t ollama-webui .
docker run -d -p 3000:8080 --name ollama-webui --restart always ollama-webui
```

Your Ollama Web UI should now be hosted at [http://localhost:3000](http://localhost:3000). Enjoy! 😄

#### Connecting to Ollama on a Different Server

If Ollama is hosted on a server other than your local machine, change `OLLAMA_API_BASE_URL` to match:

```bash
docker build --build-arg OLLAMA_API_BASE_URL='https://example.com/api' -t ollama-webui .
docker run -d -p 3000:8080 --name ollama-webui --restart always ollama-webui
```

## How to Build for Static Deployment

1. Install `node`

   ```sh
   # Mac, Linux
   curl https://webi.sh/node@lts | sh
   source ~/.config/envman/PATH.env
   ```

   ```pwsh
   # Windows
   curl.exe https://webi.ms/node@lts | powershell
   ```

2. Clone & Enter the project
   ```sh
   git clone https://github.com/ollama-webui/ollama-webui.git
   pushd ./ollama-webui/
   ```
3. Create and edit `.env`
   ```sh
   cp -RPp example.env .env
   ```
4. Run in dev mode, or build the site for deployment
   - Test in Dev mode:
     ```sh
     npm run dev
     ```
   - Build for Deploy: \
     (`PUBLIC_API_BASE_URL` will overwrite the value in `.env`)
     ```sh
     PUBLIC_API_BASE_URL='https://example.com/api' npm run build
     ```
5. Test the build with `caddy` (or the server of your choice)

   ```sh
   curl https://webi.sh/caddy | sh

   PUBLIC_API_BASE_URL='https://localhost/api' npm run build
   caddy run --envfile .env --config ./Caddyfile.localhost
   ```

## What's Next? 🚀

### To-Do List 📝

Here are some exciting tasks on our to-do list:

- 🌐 **Web Browser Extension**: Seamlessly integrate our services into your browsing experience with our convenient browser extension.
- 🚀 **Integration with Messaging Platforms**: Explore possibilities for integrating with popular messaging platforms like Slack and Discord.
- 🎨 **Customization**: Tailor your chat environment with personalized themes and styles.
- 📥🗑️ **Download/Delete Models**: Easily acquire or remove models directly from the web UI.
- 📚 **Enhanced Documentation**: Elevate your setup and customization experience with improved, comprehensive documentation.
- 🌟 **User Interface Enhancement**: Elevate the user interface to deliver a smoother, more enjoyable interaction.
- 🧐 **User Testing and Feedback Gathering**: Conduct thorough user testing to gather insights and refine our offerings based on valuable user feedback.

Feel free to contribute and help us make Ollama Web UI even better! 🙌

## Contributors ✨

A big shoutout to our amazing contributors who have helped make this project possible! 🙏

- [Ollama Team](https://github.com/jmorganca/ollama)
- [Timothy J. Baek](https://github.com/tjbck)
- [AJ ONeal](https://github.com/coolaj86)

## License 📜

This project is licensed under the [MIT License](LICENSE) - see the [LICENSE](LICENSE) file for details. 📄

## Support 💬

If you have any questions, suggestions, or need assistance, please open an issue or join our [Discord community](https://discord.gg/ollama) to connect with us! 🤝

---

Let's make Ollama Web UI even more amazing together! 💪
