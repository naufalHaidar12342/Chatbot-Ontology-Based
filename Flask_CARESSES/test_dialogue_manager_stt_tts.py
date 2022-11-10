import requests
import os
import speech_recognition as sr
import pyttsx3
import deepspeech
import mic_streaming
import logging
import numpy as np
from halo import Halo

# Location of the API (the server it is running on)
localhost = "127.0.0.1"
#azure_server = "13.95.222.73"
#cineca = "131.175.198.134"
#BASE = "http://" + cineca + ":5000/"
BASE = "http://" + localhost + ":5000/"

# initialize speech recognizer
model_path = "D:\ontology\Mentore-Group-Project\Model\deepspeech-0.8.2-models.pbmm"
model = deepspeech.Model(model_path)
scorer_path = "D:\ontology\Mentore-Group-Project\Model\deepspeech-0.8.2-models.scorer"
model.enableExternalScorer(scorer_path)
# load_scorer = dps.enableExternalScorer("D:\ontology\Mentore-Group-Project\Model\deepspeech-0.9.3-models.scorer")

# Stream from microphone to DeepSpeech using VAD

# initialize Text-to-speech engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')       #getting details of current voice
engine.setProperty('rate', 170)     # setting up new voice rate
engine.setProperty('voice', voices[1].id)

client_id = -1
if os.path.exists("credentials.txt"):
    with open("credentials.txt", 'r') as f:
        client_id = f.readline()
    text = "Welcome back client", str(client_id), "!"
    print("Welcome back client", str(client_id), "!")
    engine.say(text)
    engine.runAndWait()
    text = "I missed you. What would you like to talk about?"
    print("R: I missed you. What would you like to talk about?")
    engine.say(text)
    engine.runAndWait()


def main():

    # with sr.Microphone() as source:

        global client_id
        if client_id == -1:
            response = requests.put(BASE + "caresses/0/0", verify=False)
            client_id = response.json()['id']
            with open("credentials.txt", 'w') as credentials:
                credentials.write(str(client_id))
            print("Hey, you're new! Welcome, your ID is:", str(client_id))
            print("R:", response.json()['reply'])
        while 1:
            # read the audio data from the default microphone
            # dp = mic_streaming()
            vad_audio = mic_streaming.VADAudio(
                        input_rate=16000)
            print("Listening (ctrl-C to exit)...")
            frames = vad_audio.vad_collector()

            spinner = Halo(text='U: ', spinner='dots')
            stream_context = model.createStream()
            for frame in frames:
                if frame is not None:
                    if spinner: spinner.start()
                    logging.debug("streaming frame")
                    stream_context.feedAudioContent(np.frombuffer(frame, np.int16))
                else:
                    if spinner: spinner.stop()
                    logging.debug("end utterence")
                    sentence = stream_context.finishStream()
                    print('U:', sentence)
                    if sentence == "":
                        continue
                    break
                    # stream_context = model.createStream()

           

            
            sentence = sentence.replace(" ", "_")
            response = requests.get(BASE + "caresses/" + str(client_id) + "/" + sentence, verify=False)
            # print("Response time: ", response.elapsed.total_seconds())
            intent_reply = response.json()['intent_reply']
            reply = response.json()['reply']
            plan = response.json()['plan']
            reply = intent_reply + " " + plan + " " + reply
            print("R:", reply)
            engine.say(reply)
            engine.runAndWait()


if __name__ == '__main__':
    main()
