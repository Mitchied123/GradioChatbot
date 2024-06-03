import gradio as gr
from pathlib import Path
from openai import OpenAI
import time
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import os

load_dotenv()  

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

message_history = ['ok']  

context_length = int

#get a time a make it a nice format
now = time.time()
formatted_now = time.ctime(now)
formatted_now = formatted_now.replace(":", "_")
#variables for holding making chat log names
ChatLog = f"Chat History {formatted_now}"
ContextLog =f"Chat Contextualistion {formatted_now}"


def contextualise(response_text):
    context = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system","content": "For AI contextual awareness create a perfect output history log the following information in a JSON data structure, do not preface the information respond only with a JSON and no other characters like quotation marks or apostrophies"},
                {"role": "user", "content": response_text},
            ],
            max_tokens=500,
        )
    contextS = context.choices[0].message.content
    message_history.append(contextS)
    
    #output contextualisation history
    with open(f"{ContextLog}.txt", "a") as file:
        file.write("Contextualisation " + contextS + "\n")

    print('contextualised')
    
def reflect(message_history):
    if len(message_history) > 9:
        reflection = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system","content": "For AI contextual awareness here are the previous responses, consolidate the following information in about 300 words, create a perfect output history log of that information in a JSON data structure, do not preface the information respond only with a JSON and no other characters like quotation marks or apostrophies"},
                {"role": "user", "content": message_history},
            ],
            max_tokens=500,
        )
        message_history = reflection.choices[0].message.content
        print("reflected")

        with open(f"{ContextLog}.txt", "a") as file:
            file.write("Reflection " + message_history + "\n")
        return message_history
    else:
        print(len(message_history))
        return(message_history)

def chitchat(instructions, message_history, transcript_text, response_seconds, response_tokens):
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system","content": instructions},
                {"role" : "assistant", "content": f"For contextual awareness here are the previous responses{message_history}"},
                {"role": "user", "content": transcript_text},
                {"role": "user", "content": f"Make the response length {response_seconds} WORDS long"},
            ],
            max_tokens=response_tokens,
        )
        response_text = response.choices[0].message.content
        
        #output chat history
        with open(f"{ChatLog}.txt", "a") as file:
            file.write("ASSISTANT: " + response_text + "\n")
        
        return response_text
    
    except Exception as e:
        print(f"Error during chat response: {e}")
        return None, f"Error during chat response: {e}"

def tts(response_text, transcript_text, voice_model):
    try:
        speech_file_path = Path(__file__).parent / "speech.mp3"
        response_audio = client.audio.speech.create(
            model="tts-1",
            voice=voice_model,
            input=response_text,
        )
        response_audio.stream_to_file(speech_file_path)
        #puts user speech in a log

        with open(f"{ChatLog}.txt", "a") as file:
            file.write("USER: " + transcript_text + "\n ")

        return str(speech_file_path)
    except Exception as e:
        print(f"Error during TTS generation: {e}")
        return None, response_text, transcript_text, None

def greet(instructions, Response_Length, inAudio, voice_model):
    global message_history
    # Read audio file
    try:
        with open(inAudio, "rb") as audio_file:
            start_transcription = time.time()  # Timestamp before transcription
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
            transcript_text = transcript.text
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None, "Error during transcription", None, None
    
    def response_time_manager(response_length):
        response_seconds = int(response_length * 3)
        response_tokens = int(response_length * 3 * 1.34)
        return response_seconds, response_tokens

    response_seconds, response_tokens = response_time_manager(Response_Length)

    # Generate response from chat model
    response_text = chitchat(instructions, message_history, transcript_text, response_seconds, response_tokens)


    with ThreadPoolExecutor() as executor:
        # Generate response audio
        future1 = executor.submit(tts, response_text, transcript_text, voice_model)
        # Generate history of chat
        future2 = executor.submit(contextualise, response_text)
        #wait till its done
        future2.result()
        #reflect on previous responses and summarise message histor
        future3 = executor.submit(reflect, message_history)

    speech_file_path = future1.result()

    message_history = future3.result()

    return str(speech_file_path), response_text, transcript_text, message_history

input_audio = gr.Audio(
    sources=["microphone"],
    type="filepath")

length = gr.Slider(
    10,150, 
    value = 60,
    label = "Response Length (Seconds)"
)

voice_model = gr.Dropdown(
    ["alloy", "echo", "fable", "onyx", "nova","shimmer"],
    label="Voice Model",
    value="nova"
)

instructions = gr.Textbox(
    value="You are a super intelligent computer designed to educate",
    lines=3,
    label="Instructions"
)

demo = gr.Interface(
    fn=greet,
    inputs=[instructions, length, input_audio, voice_model],
    outputs=["audio", "text", "text", "text"]
)

demo.launch()
