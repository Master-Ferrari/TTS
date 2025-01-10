# example.py
import sys
import torch
from TTS.api import TTS
import io
import json
import time

def decode_from_numbers(encoded_text):
    numbers = encoded_text.split(',')
    decoded_text = ''.join(chr(int(num)) for num in numbers)
    return decoded_text

def encode_to_numbers(decoded_text):
    encoded_text = ','.join(str(ord(char)) for char in decoded_text)
    return encoded_text

def read_JSON(json_string: str):
    try:
        parsed_json = json.loads(json_string)
        prompt = parsed_json['prompt']
        voice = parsed_json['voice']
        file_path = parsed_json['file']
        return prompt, voice, file_path
    except (json.JSONDecodeError, KeyError) as e:
        print(f'Error parsing JSON or missing key: {e}')
        return None, None, None


def print_to_node(text):
    print(encode_to_numbers(" \n" + text + " "))
    sys.stdout.flush()




sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

print_to_node("ready!")

for line in sys.stdin:
    input_text = decode_from_numbers(line.strip())
    
    print_to_node("input from node:    " + input_text)

    prompt, voice, path = read_JSON(input_text)
    voice = f"..\\misc\\audio\\voices\\{voice}.wav"

    tts.tts_to_file(text=prompt, speaker_wav=voice, language="ru", file_path=path)

    print_to_node("wav is done!")
