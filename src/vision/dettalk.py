import cv2
import dlib
import time
import logging
import numpy as np
from ctransformers import AutoModelForCausalLM

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("eye-status.log")],
)

def calc_eye(eye):
    a = np.linalg.norm(eye[1] - eye[5])
    b = np.linalg.norm(eye[2] - eye[4])
    c = np.linalg.norm(eye[0] - eye[3])
    return (a + b) / (2 * c)

def draw_eyes(frame, eye):
    cv2.polylines(frame, [eye], isClosed=True, color=(0, 255, 0), thickness=1)

face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("../../data/shape-pred-face-landmarks.dat")

cap = cv2.VideoCapture(0)

eyes_detected = False
eyes_closed_timer = 0
conversation_start_time = None
chatbot_active = False
chatbot_start_time = None
chatbot_duration = 300  # 5 minutes

# Load the chat model
chat_model = AutoModelForCausalLM.from_pretrained(
    'llama-2-7b-chat.ggmlv3.q8_0.bin',
    model_type='llama',
    temperature=0.1, 
    top_p=0.9,
    max_new_tokens = 1000,
    context_length=6000
)

messages = [{"role": "assistant", "content": "How may I assist you today?"}]

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

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    if not chatbot_active:
        for face in faces:
            landmarks = landmark_predictor(gray, face)
            left_eye = np.array(
                [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            )
            right_eye = np.array(
                [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
            )

            ear_avg = (calc_eye(left_eye) + calc_eye(right_eye)) / 2

            if ear_avg < 0.2:
                eyes_closed_timer += 1
                if eyes_closed_timer > 5:
                    logging.debug("Eyes Closed")
                    # Trigger chatbot
                    chatbot_active = True
                    chatbot_start_time = time.time()
                    print("Are you feeling sleepy? I can tell you a short story. Lemme think...")
                    user_input = "I am feeling sleepy. Can you tell me a short story in under 500 characters?"
                    response = generate_response(user_input, chat_model)
                    print("Assistant:", response)
                    messages.append({"role": "user", "content": user_input})
                    messages.append({"role": "assistant", "content": response})
                    eyes_detected = False
                    break
            else:
                eyes_closed_timer = 0

            # DEBUG: eye lines display
            draw_eyes(frame, left_eye)
            draw_eyes(frame, right_eye)

    if chatbot_active:
        if time.time() - chatbot_start_time > chatbot_duration:
            chatbot_active = False
            conversation_start_time = time.time()

    if not chatbot_active and conversation_start_time and time.time() - conversation_start_time > 600:
        eyes_detected = True

    cv2.imshow("alert drive", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Check for user input to stop talking or continue conversation
    if chatbot_active:
        user_input = input("You: ")
        messages.append({"role": "user", "content": user_input})
        if user_input.lower() == "stop, i'm not sleepy anymore":
            chatbot_active = False
        else:
            response = generate_response(user_input, chat_model)
            print("Assistant:", response)
            messages.append({"role": "assistant", "content": response})
            conversation_start_time = time.time()

cap.release()
cv2.destroyAllWindows()
