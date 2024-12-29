import streamlit as st
import requests

# NOTE: ollama must be running for this to work, start the ollama app or run `ollama serve`
model = "llama3.2"  # TODO: update this for whatever model you wish to use

# Define the system prompt
system_prompt = "You are a helpful assistant."

def chat_with_llama(prompt):
    try:
        full_prompt = f"{system_prompt}\n\nUser: {prompt}"
        
        payload = {
            "model": model,  
            "prompt": full_prompt,    
            "stream": False      
        }
        response = requests.post('http://127.0.0.1:11434/api/generate', json=payload)
        response.raise_for_status() 
        data = response.json()
        return data.get("response", "No response from the model.")
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

def main():
    st.title('Ollama Chat')
    st.write(f"Welcome to the Ollama Chat! Ask me anything. You are running the `{model}` model.")
    st.markdown(
    """
    **Command Available:**  
    - `clear` - clear chat history
    """
    )
    # chat
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Enter your message")

    if user_input:
        if user_input.lower() in ['clear', 'reset']:
            st.session_state.chat_history = []
            st.rerun()  
        elif user_input.lower() in ['list']:
            st.info(f"Current model: {model}", icon="ℹ️")
        else:
            st.session_state.chat_history.append({"role": "User", "content": user_input})

            with st.chat_message("User"):
                st.write(user_input)

            response = chat_with_llama(user_input)

            st.session_state.chat_history.append({"role": "Model", "content": response})

            with st.chat_message("Model"):
                st.write(response)

if __name__ == "__main__":
    main()