<<<<<<< HEAD
# 🧠 OpenAI Quickstart Project (Flask + API)

This project is a simple web application built using Flask and the OpenAI API. It allows users to submit a prompt and receive an AI-generated response.


<!-- Badges -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)

---

## 📋 Table of Contents

- [About](#about)  
- [Initial Setup](#initial-setup)  
- [Current Features](#current-features)  
- [Contributing](#contributing)  
- [Author](#author)  

---

## About

This project is a simple web app built with Flask and OpenAI API. It lets users send prompts and get AI responses.

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository  
2. Create your feature branch (`git checkout -b feature-name`)  
3. Commit your changes (`git commit -m 'Add feature'`)  
4. Push to the branch (`git push origin feature-name`)  
5. Open a Pull Request  

Please follow the existing code style and write clear commit messages.

## 🛠️ Initial Setup

Below are all the steps I followed manually to configure the environment. This demonstrates mastery of essential web development tools and AI integration.

1. Clone the repository

git clone https://github.com/your-username/openai-quickstart-01.git
cd openai-quickstart-01

2. Create and activate the virtual environment  
Windows:  
python -m venv venv  
venv\Scripts\activate  

Linux/Mac:  
python3 -m venv venv  
source venv/bin/activate  

3. Install dependencies  
pip install -r requirements.txt  

If you don't have a requirements.txt, create it with:  
pip freeze > requirements.txt  

Make sure it includes at least:  
flask  
openai  
python-dotenv  

4. Configure the OpenAI API key  
Create a .env file with the following content:  
OPENAI_API_KEY=your-api-key-here  

Important: Use .gitignore to never upload your key to GitHub.

5. Expected basic structure  
openai-quickstart-01/  
├── app.py  
├── .env  
├── requirements.txt  
├── README.md  
├── templates/  
│   └── index.html  
├── static/  
│   └── css/  
│       └── style.css  

6. Run the app locally  
flask run  

Open your browser at: http://127.0.0.1:5000

---

## 📌 Current Features

- Web interface with a prompt submission form  
- Integration with OpenAI API  
- Basic error handling  
- User input length limit  
- Simple CSS styling for frontend  

---

## 👨‍💻 Author

Lucas Alencar  
ADS student, mobile-first beginner developer, passionate about Python, AI, and transformative technologies.  
GitHub: https://github.com/lucasalencarxisto-stack



---
=======
# -Training-ChatGPT-to-Recognize-Custom-Keywords-
Train ChatGPT to recognize custom keywords and simulate memory using prompt engineering. A personal semantic tagging system for your conversations.
# 🧠 Training ChatGPT to Recognize Custom Keywords  
> Empowering conversations through memory and semantic tags.

![GitHub Repo stars](https://img.shields.io/github/stars/lucasalencarxisto-stack/chatgpt-keyword-memory?style=social)
![License](https://img.shields.io/badge/license-MIT-green)
![Made with](https://img.shields.io/badge/Made%20with-❤️%20and%20GPT-blue)

---

## 📌 Project Description

This project demonstrates how to **train ChatGPT to recognize and respond to custom keywords** (a.k.a. "semantic bookmarks") in natural conversation. Using simple prompts and context memory, you can simulate an intelligent tagging system — similar to how developers use variables, or how your brain retrieves associated ideas when hearing a trigger word.

✨ **Why this matters:**  
As AI assistants evolve, building systems that understand *your unique way of thinking* becomes crucial. This project showcases how to build a *personal AI-enhanced memory* by teaching ChatGPT to associate specific concepts, notes, or documents with keywords you define — creating an illusion of persistent memory.

---

## 🧪 How It Works

- You define a custom keyword (e.g., `openai-quickstart-01`, `Ideia_portifólio.01`)
- ChatGPT is instructed to remember and recognize that keyword when mentioned
- The keyword links to a piece of content or context from a past interaction
- You can later retrieve that information just by typing the keyword

---

## 🛠️ Use Cases

- 🗂️ Creating personal knowledge bases  
- 🧭 Building navigation systems within long conversations  
- 🧠 Externalizing memory for brainstorming and study  
- 🧩 Managing multiple projects, notes, or documents  

---

## 📚 Example Prompt

```text
Salvar palavra-chave: openai-quickstart-01

🌍 Multilingual Awareness
This project also includes translations of the full concept and documentation in:

🇧🇷 Portuguese

🇪🇸 Spanish

🇨🇳 Mandarin Chinese

(See /translations folder or the bottom of this README)

🤖 Future Goals
 Build a plugin or browser extension for ChatGPT to automate keyword tagging

 Create a web interface to manage stored tags and values

 Integrate with GitHub for saving and syncing memory contexts

👨‍💻 Author
Lucas Alencar
Student of Systems Analysis and Development | Aspiring OpenAI Dev | Passionate about AI and Memory-Augmented Interfaces
🔗 GitHub Profile: https://github.com/lucasalencarxisto-stack

📝 License
MIT License — use freely, learn openly, evolve collaboratively.

🌐 Translations
Click below to view full documentation in other languages:

🇧🇷 Versão em Português

🇪🇸 Versión en Español

🇨🇳 中文版本

(Coming soon...)

# 🧠 Training ChatGPT to Recognize Custom Keywords

A semantic keyword system to simulate memory and create a personal assistant experience using ChatGPT. This project demonstrates how to create a custom tagging system with keywords for easy context retrieval and project organization.

## 📌 Table of Contents
- [🌐 English](#-english)
- [🇧🇷 Português](#-português)
- [🇪🇸 Español](#-español)
- [🇨🇳 中文 (Mandarim)](#-中文-mandarim)

---

## 🌐 English

### 🧠 Project Overview

This experiment shows how to train ChatGPT to recognize and retrieve custom keywords, creating a kind of memory simulation through prompt engineering. It allows users to associate code names with detailed context (projects, goals, documents, ideas), making future access easier and faster.

### 🛠️ How It Works

- You define a **keyword** and give ChatGPT the context to remember.
- Later, simply call the keyword, and the AI will retrieve the associated info.
- It’s like building your personal assistant inside ChatGPT, without plugins.

### 🧪 Example

```text
User: Save the keyword "openai-quickstart-01"  
ChatGPT: ✅ Keyword "openai-quickstart-01" saved!

User: openai-quickstart-01  
ChatGPT: [Returns full context saved earlier]

📚 Use Cases
Project tracking

Learning logs

Brainstorm organization

Personal knowledge base

🇪🇸 Español
🧠 Descripción del Proyecto
Este experimento muestra cómo entrenar a ChatGPT para reconocer y recuperar palabras clave personalizadas, simulando una especie de memoria mediante ingeniería de prompts. El usuario puede asociar palabras clave a contextos importantes (proyectos, metas, documentos, ideas) para acceder a ellos fácilmente más tarde.

🛠️ Cómo Funciona
Defines una palabra clave y proporcionas el contexto a ChatGPT.

Luego, simplemente llamas a la palabra clave para recuperar la información.

Es como tener un asistente personal dentro de ChatGPT, sin necesidad de plugins.

🧪 Ejemplo
Usuario: Guardar palabra clave "openai-quickstart-01"  
ChatGPT: ✅ Palabra clave "openai-quickstart-01" guardada.

Más tarde...
Usuario: openai-quickstart-01  
ChatGPT: [Devuelve el contexto guardado anteriormente]

📚 Casos de Uso
Seguimiento de proyectos

Registros de aprendizaje

Organización de ideas

Base de conocimiento personal

🇨🇳 中文 (Mandarim)
🧠 项目概述
这个实验展示了如何训练 ChatGPT 识别和检索自定义关键词，通过提示工程模拟“记忆”系统。用户可以将关键字与重要的上下文（项目、目标、文档、想法）关联起来，便于日后快速访问。

🛠️ 工作原理
你定义一个关键词，并告诉 ChatGPT 相关内容。

以后只需输入关键词，AI 就会返回相关信息。

就像在 ChatGPT 内部构建了一个个人助理，无需插件。

🧪 示例
用户：保存关键词 "openai-quickstart-01"  
ChatGPT：✅ 关键词 "openai-quickstart-01" 已保存！

稍后...
用户：openai-quickstart-01  
ChatGPT：[返回之前保存的内容]

📚 应用场景
项目管理

学习日志

思维整理

个人知识库

🧠 Inspired by
My journey to becoming a junior developer and aiming to collaborate with OpenAI.
Follow me on GitHub and check my portfolio in progress!🧠 Inspired by
My journey to becoming a junior developer and aiming to collaborate with OpenAI.
Follow me on GitHub and check my portfolio in progress!



>>>>>>> 9e80b4bbf089282759a79e2e128702b079880881


