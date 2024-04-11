import os
from ctransformers import AutoModelForCausalLM


def load_chat_model(temperature, top_p):
    return AutoModelForCausalLM.from_pretrained(
        # 'models/llama-2-7b-chat.ggmlv3.q2_K.bin', 
        'llama-2-7b-chat.ggmlv3.q8_0.bin',
        model_type='llama',
        temperature=temperature, 
        top_p=top_p
    )

def generate_response(user_input, chat_model):
    name = "rahul"
    string_dialogue = f"You are a helpful assistant. You help drivers stay alert. The driver's name is {name}. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    
    # Append previous messages
    for message in messages:
        if message["role"] == "user":
            string_dialogue += f"User: {message['content']}\\n\\n"
        else:
            string_dialogue += f"Assistant: {message['content']}\\n\\n"
    
    # Generate response
    output = chat_model(f"prompt {string_dialogue} {user_input} Assistant: ")
    return output

# Load the chat model
chat_model = load_chat_model(0.1, 0.9)

# Maintain chat history
messages = [{"role": "assistant", "content": "How may I assist you today?"}]

def clear_chat_history():
    global messages
    messages = [{"role": "assistant", "content": "How may I assist you today?"}]

def main():
    clear_chat_history()
    print("Welcome to the Alert-Drive Chatbot!")
    print("You are now chatting with the assistant. Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        
        # Generate response
        response = generate_response(user_input, chat_model)
        print("Assistant:", response)

if __name__ == "__main__":
    main()
