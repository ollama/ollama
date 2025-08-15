document.addEventListener('DOMContentLoaded', () => {
    const sendButton = document.getElementById('send-button');
    const messageInput = document.getElementById('message-input');
    const chatWindow = document.getElementById('chat-window');
    const modelSelector = document.getElementById('model-selector');
    const crewSelector = document.getElementById('crew-selector');
    const thinkingIndicator = document.getElementById('thinking-indicator');

    let messages = [];

    const addMessage = (sender, text) => {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', `${sender}-message`);
        messageElement.innerText = text;
        chatWindow.appendChild(messageElement);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    };

    const sendMessage = async () => {
        const messageText = messageInput.value.trim();
        if (messageText === '') {
            return;
        }

        addMessage('user', messageText);
        messageInput.value = '';

        messages.push({ role: 'user', content: messageText });

        thinkingIndicator.style.display = 'block';

        const selectedCrew = crewSelector.value;
        let endpoint = '/chat';
        let body = {};

        if (selectedCrew === 'chat') {
            endpoint = '/chat';
            body = {
                model: modelSelector.value,
                messages: messages,
            };
        } else {
            endpoint = '/crew';
            body = {
                crew: selectedCrew,
                topic: messageText,
            };
        }

        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(body),
            });

            thinkingIndicator.style.display = 'none';

            if (!response.ok) {
                const errorData = await response.json();
                addMessage('assistant', `Error: ${errorData.error}`);
                return;
            }

            const data = await response.json();
            let assistantMessage;
            if (selectedCrew === 'chat') {
                assistantMessage = data.message.content;
            } else {
                assistantMessage = data.message;
            }

            addMessage('assistant', assistantMessage);
            messages.push({ role: 'assistant', content: assistantMessage });

        } catch (error) {
            thinkingIndicator.style.display = 'none';
            addMessage('assistant', `Error: ${error.toString()}`);
        }
    };

    sendButton.addEventListener('click', sendMessage);
    messageInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });
});
