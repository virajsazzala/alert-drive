import os
import pygame
import logging
from consts import VOICE_DATA
import speech_recognition as sr


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("speech-status.log")],
)

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
