import cv2
import dlib
import time
import logging
import numpy as np
from ctransformers import AutoModelForCausalLM

import re
import os
import pygame
import logging
from consts import VOICE_DATA
import speech_recognition as sr

from vosk import Model, KaldiRecognizer
import pyaudio
from consts import VOICE_MODEL

r = sr.Recognizer()


def say(audio):
    voice = "en-AU-NatashaNeural"
    command = (
        f'edge-tts --voice "{voice}" --text "{audio}" --write-media "{VOICE_DATA}"'
    )
    os.system(command)

    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load(VOICE_DATA)

    try:
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        logging.critical(e)
    finally:
        pygame.mixer.music.stop()
        pygame.mixer.quit()


def listen():
    print("Listening...")
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        r.pause_threshold = 1
        audio = r.listen(source)

        try:
            # NOTE: use recognize_google for faster but less accurate recognition. (set = language="en-US")
            # NOTE: use recognize_whisper for slower but accurate recognition.
            text = r.recognize_whisper(audio)
            logging.info(f"you said: {text}")

            return text.lower()
        except Exception as e:
            raise Exception(str(e))


def lis():
    mp = str(VOICE_MODEL.resolve())
    model = Model(mp)
    recognizer = KaldiRecognizer(model, 16000)

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=8000,
    )
    stream.start_stream()

    while True:
        data = stream.read(8000)
        if recognizer.AcceptWaveform(data):
            said = recognizer.Result()[14:-3]
            if len(said) == 0:
                break
            print(said)
            return said


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
landmark_predictor = dlib.shape_predictor("../data/shape-pred-face-landmarks.dat")

cap = cv2.VideoCapture(0)

eyes_detected = False
eyes_closed_timer = 0
conversation_start_time = None
chatbot_active = False
chatbot_start_time = None
chatbot_duration = 300  # 5 minutes

# Load the chat model
chat_model = AutoModelForCausalLM.from_pretrained(
    "vision/llama-2-7b-chat.ggmlv3.q8_0.bin",
    model_type="llama",
    temperature=0.1,
    top_p=0.9,
    max_new_tokens=1000,
    context_length=6000,
)

messages = [{"role": "assistant", "content": "How may I assist you today?"}]


def generate_response(user_input, chat_model):
    name = "rahul"
    string_dialogue = f"You are a helpful assistant. You help drivers stay alert. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."

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
                    say(
                        "Are you feeling sleepy? I can tell you a short story. Lemme think of one."
                    )
                    user_input = "I am feeling sleepy. Can you tell me a short story, not about driving?"
                    # response = generate_response(user_input, chat_model)
                    response = "a cat was running in the garden"
                    say(response.replace("\ n \ n", " "))
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

    if (
        not chatbot_active
        and conversation_start_time
        and time.time() - conversation_start_time > 600
    ):
        eyes_detected = True

    cv2.imshow("alert drive", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if chatbot_active:
        # user_input = input("You: ")
        user_input = lis()
        messages.append({"role": "user", "content": user_input})
        if re.search(r"\bstop\b.*\bnot\s+sleepy\b", user_input.lower(), re.IGNORECASE):
            chatbot_active = False
            say("Okay, I'm turning off now!.")
            break
        else:
            response = generate_response(user_input, chat_model)
            say(response.replace("\ n \ n", " "))
            messages.append({"role": "assistant", "content": response})
            conversation_start_time = time.time()

cap.release()
cv2.destroyAllWindows()
