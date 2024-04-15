from vosk import Model, KaldiRecognizer
import pyaudio
from consts import VOICE_MODEL

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