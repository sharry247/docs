from flask import Flask, request, jsonify, send_file, Response
import json
import random
import string
import threading
import time
from typing import Any
from flask import Flask, request, jsonify
import json
from flask import Flask, request, jsonify
import argparse
import requests
import json
import logging
from webscout import transcriber
import speech_recognition as sr
from flask import Flask, request, jsonify
import argparse
from gradio_client import Client, file
import re
from generator import image_generator
from typing import Optional
from webscout import WEBS
from typing import List, Dict, Union
import datetime
import g4f
from fp.fp import FreeProxy
from g4f import ChatCompletion
from g4f.Provider import GptGo
from loguru import logger
from webscout import play_audio
app = Flask(__name__)

API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
headers = {"Authorization": "Bearer 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"}
AI_URL = "https://api-inference.huggingface.co/models/dima806/facial_emotions_image_detection"
haders = {"Authorization": "Bearer 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"}

def EmotionDetector(filename):
    """Detects emotions in an image using the Hugging Face API."""
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(AI_URL, haders=headers, data=data)
    return response.json()




# thread local proxy variable
thread_local = threading.local()
thread_local.proxy = None
free_proxy = FreeProxy(timeout=1)


class TranscriberYT:
    def __init__(self):
        pass

    def extract_transcript(self, video_id):
        """Extracts the transcript from a YouTube video."""
        try:
            transcript_list = transcriber.list_transcripts(video_id)
            for transcript in transcript_list:
                transcript_data_list = transcript.fetch()
                lang = transcript.language
                transcript_text = ""
                if transcript.language_code == 'en':
                    for line in transcript_data_list:
                        start_time = line['start']
                        end_time = start_time + line['duration']
                        formatted_line = f"{start_time:.2f} - {end_time:.2f}: {line['text']}\n"
                        transcript_text += formatted_line
                    return transcript_text
                elif transcript.is_translatable:
                    english_transcript_list = transcript.translate('en').fetch()
                    for line in english_transcript_list:
                        start_time = line['start']
                        end_time = start_time + line['duration']
                        formatted_line = f"{start_time:.2f} - {end_time:.2f}: {line['text']}\n"
                        transcript_text += formatted_line
                    return transcript_text
            return "Transcript extraction failed. Please check the video URL."
        except Exception as e:
            return f"Error: {e}"


def generate_images(filename='', number=1, prompt='RANDOM', prompt_size=10, negative_prompt='nudity text', style='RANDOM', resolution='512x768', guidance_scale=7):
    # Validate resolution format
    resolution_pattern = r'\d{2,4}x\d{2,4}'
    if not re.match(resolution_pattern, resolution):
        raise ValueError('Invalid resolution formatting. Example: "512x768".')

    # Configure the image generator
    generator = image_generator(
        base_filename=filename,
        amount=number,
        prompt=prompt,
        prompt_size=prompt_size,
        negative_prompt=negative_prompt.replace(' ', ', '),
        style=style,
        resolution=resolution,
        guidance_scale=guidance_scale
    )

    # Generate images
    for _ in generator:
        pass

def get_proxy(is_working: bool = True):
    if is_working:
        if getattr(thread_local, "proxy") is None:
            setattr(thread_local, "proxy", free_proxy.get())
        else:
            return thread_local.proxy
    else:
        setattr(thread_local, "proxy", free_proxy.get())
        return thread_local.proxy


def log_api_call(request_data, response):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("api_calls_log.txt", "a") as log_file:
        log_file.write(f"{timestamp}\n")
        log_file.write(f"Request Data: {request_data}\n")
        log_file.write(f"Response: {response}\n\n")
class LLM:
    def __init__(self, model: str, system_message: str = "You are a Helpful AI."):
        self.model = model
        self.conversation_history = [{"role": "system", "content": system_message}]

    def chat(self, messages: List[Dict[str, str]], max_token=None) -> Union[str, None]:
        url = "https://api.deepinfra.com/v1/openai/chat/completions"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Accept-Language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Content-Type': 'application/json',
            'Origin': 'https://deepinfra.com',
            'Pragma': 'no-cache',
            'Referer': 'https://deepinfra.com/',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-site',
            'X-Deepinfra-Source': 'web-embed',
            'accept': 'text/event-stream',
            'sec-ch-ua': '"Google Chrome";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"'
        }
        data = json.dumps(
            {
                'model': self.model,
                'messages': messages,
                'temperature': 0.7,
                'max_tokens': max_token,
                'stop': [],
                'stream': False #dont change it
            }, separators=(',', ':')
        )
        try:
            result = requests.post(url=url, data=data, headers=headers)
            return result.json()['choices'][0]['message']['content']
        except:
            return None
    

def GenerativeIO(user_input,model,max_tokens,system_prompt):
    llm = LLM(model=model, system_message=system_prompt)
    messages = [
        {"role": "user", "content": user_input}
    ]
    response = llm.chat(messages,max_token=max_tokens)
    return response
        
# List of supported models
supported_models = [
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "microsoft/WizardLM-2-8x22B",
    "microsoft/WizardLM-2-7B",
    "Mixtral-8x22B-v0.1",
    "WizardLM-2 8x22B",
    "WizardLM-2 7B",
    "gpt2",
    "HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "databricks/dbrx-instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "meta-llama/Llama-2-70b-chat-hf",
    "cognitivecomputations/dolphin-2.6-mixtral-8x7b",
    "lizpreciatior/lzlv_70b_fp16_hf",
    "openchat/openchat_3.5",
    "mistralai/Mixtral-8x22B-v0.1",
    "meta-llama/Llama-2-7b-chat-hf",
    "Austism/chronos-hermes-13b-v2",
    "EleutherAI/gpt-j-6B",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-125M",
    "EleutherAI/gpt-neo-2.7B",
    "EleutherAI/pythia-12b",
    "EleutherAI/pythia-2.8b",
    "bigcode/starcoder",
    "codellama/CodeLlama-34b-Instruct-hf",
    "codellama/CodeLlama-70b-Instruct-hf"
]

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    user_input = data.get('user_input')
    model = data.get('model')
    max_tokens = data.get('max_tokens')
    system_prompt = data.get('system_prompt')

    response = GenerativeIO(user_input, model, max_tokens, system_prompt)
    
    # Log the API call
    request_data = {
        'user_input': user_input,
        'model': model,
        'max_tokens': max_tokens,
        'system_prompt': system_prompt
    }
    log_api_call(request_data, response)
    
    return jsonify({'response': response})

@app.route('/models', methods=['GET'])
def get_models():
    return jsonify({'models': supported_models})


@app.route("/chat/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    req_data = request.get_json()
    stream = req_data.get("stream", False)
    model = req_data.get("model", "gpt-3.5-turbo")
    messages = req_data.get("messages")
    temperature = req_data.get("temperature", 1.0)
    top_p = req_data.get("top_p", 1.0)
    max_tokens = req_data.get("max_tokens", 4096)

    logger.info(
        f"chat_completions: stream: {stream}, model: {model}, temperature: {temperature}, top_p: {top_p}, max_tokens: {max_tokens}"
    )

    response = gen_resp(max_tokens, messages, model, stream, temperature, top_p)

    completion_id = "".join(random.choices(string.ascii_letters + string.digits, k=28))
    completion_timestamp = int(time.time())

    if not stream:
        logger.info(f"chat_completions: response: {response}")
        return jsonify({
            "id": f"chatcmpl-{completion_id}",
            "object": "chat.completion",
            "created": completion_timestamp,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            },
        })

    def streaming():
        for chunk in response:
            completion_data = {
                "id": f"chatcmpl-{completion_id}",
                "object": "chat.completion.chunk",
                "created": completion_timestamp,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": chunk,
                        },
                        "finish_reason": None,
                    }
                ],
            }

            content = json.dumps(completion_data, separators=(",", ":"))
            yield f"data: {content}\n\n"
            time.sleep(0.1)

        end_completion_data: dict[str, Any] = {
            "id": f"chatcmpl-{completion_id}",
            "object": "chat.completion.chunk",
            "created": completion_timestamp,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        content = json.dumps(end_completion_data, separators=(",", ":"))
        yield f"data: {content}\n\n"

    return Response(streaming(), mimetype="text/event-stream")

def gen_resp(max_tokens, messages, model, stream, temperature, top_p):
    is_working = True
    while True:
        try:
            response = ChatCompletion.create(
                model=model,
                stream=stream,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                system_prompt="",
                provider=GptGo,
                proxy=get_proxy(is_working),
            )
            return response
        except Exception as e:
            logger.error(f"gen_resp: Exception: {e}")
            is_working = False

@app.route('/api/emotion-detector', methods=['POST'])
def emotion_detector_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if file:
        filename = file.filename
        file.save(filename)
        output = EmotionDetector(filename)
        return jsonify(output)


@app.route("/v1/completions", methods=["POST"])
def completions():
    req_data = request.get_json()
    model = req_data.get("model", "text-davinci-003")
    prompt = req_data.get("prompt")
    temperature = req_data.get("temperature", 1.0)
    top_p = req_data.get("top_p", 1.0)
    max_tokens = req_data.get("max_tokens", 4096)

    response = g4f.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    completion_id = "".join(random.choices(string.ascii_letters + string.digits, k=24))
    completion_timestamp = int(time.time())

    return jsonify({
        "id": f"cmpl-{completion_id}",
        "object": "text_completion",
        "created": completion_timestamp,
        "model": "text-davinci-003",
        "choices": [
            {"text": response, "index": 0, "logprobs": None, "finish_reason": "length"}
        ],
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        },
    })

def text_to_speech(message, voice="Brian"):
    # Generate audio content from the message using the specified voice
    audio_content = play_audio(message, voice=voice)
    
    # Save the audio to a file
    with open("output.mp3", "wb") as f:
        f.write(audio_content)
    return "output.mp3"

def predict_with_model(image_path, text, decoding_strategy="Greedy", temperature=0.4, max_new_tokens=512, repetition_penalty=1.2, top_p=0.8):
    # Initialize the Client object with the model identifier
    client = Client("HuggingFaceM4/idefics-8b")

    # Call the predict method with the provided parameters
    result = client.predict(
        image=file(image_path), # This can now be a local file path or a URL
        text=text,
        decoding_strategy=decoding_strategy,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        top_p=top_p,
        api_name="/model_inference"
    )

    # Return the result
    return result
TIMEOUT = 10
PROXY = None

@app.route('/api/search', methods=['GET'])
def search_text():
    query = request.args.get('q', '')
    max_results = request.args.get('max_results', 10, type=int)
    timelimit = request.args.get('timelimit', None)
    safesearch = request.args.get('safesearch', 'moderate')
    region = request.args.get('region', 'wt-wt')
    WEBS_instance = WEBS() # Instantiate WEBS without context manager
    results = []
    with WEBS() as webs:
        for result in enumerate(WEBS_instance.text(query, max_results=max_results, timelimit=timelimit, safesearch=safesearch, region=region)):
            results.append(result)

    return jsonify({'results': results})


@app.route('/api/images', methods=['GET'])
def search_images():
    query = request.args.get('q', '')
    max_results = request.args.get('max_results', 10, type=int)
    safesearch = request.args.get('safesearch', 'moderate')
    region = request.args.get('region', 'wt-wt')
    WEBS_instance = WEBS()
    results = []
    with WEBS() as webs:
        for result in enumerate(WEBS_instance.images(query, max_results=max_results, safesearch=safesearch, region=region)):
            results.append(result)

    return jsonify({'results': results})

@app.route('/api/videos', methods=['GET'])
def search_videos():
    query = request.args.get('q', '')
    max_results = request.args.get('max_results', 10, type=int)
    safesearch = request.args.get('safesearch', 'moderate')
    region = request.args.get('region', 'wt-wt')
    timelimit = request.args.get('timelimit', None)
    resolution = request.args.get('resolution', None)
    duration = request.args.get('duration', None)
    WEBS_instance = WEBS()
    results = []
    with WEBS() as webs:
        for result in enumerate(WEBS_instance.videos(query, max_results=max_results, safesearch=safesearch, region=region, timelimit=timelimit, resolution=resolution, duration=duration)):
            results.append(result)

    return jsonify({'results': results})

@app.route('/api/news', methods=['GET'])
def search_news():
    query = request.args.get('q', '')
    max_results = request.args.get('max_results', 10, type=int)
    safesearch = request.args.get('safesearch', 'moderate')
    region = request.args.get('region', 'wt-wt')
    timelimit = request.args.get('timelimit', None)
    WEBS_instance = WEBS()
    results = []
    with WEBS() as webs:
        for result in enumerate(WEBS_instance.news(query, max_results=max_results, safesearch=safesearch, region=region, timelimit=timelimit)):
            results.append(result)

    return jsonify({'results': results})

@app.route('/api/maps', methods=['GET'])
def search_maps():
    query = request.args.get('q', '')
    place = request.args.get('place', None)
    max_results = request.args.get('max_results', 10, type=int)
    WEBS_instance = WEBS()
    results = []
    with WEBS() as webs:
        for result in enumerate(WEBS_instance.maps(query, place=place, max_results=max_results)):
            results.append(result)

    return jsonify({'results': results})

@app.route('/api/translate', methods=['GET'])
def translate_text():
    query = request.args.get('q', '')
    to_lang = request.args.get('to', 'en')
    WEBS_instance = WEBS()
    with WEBS() as webs:
        translation = enumerate(WEBS_instance.translate(query, to=to_lang))

    return jsonify({'translation': translation})

@app.route('/api/suggestions', methods=['GET'])
def search_suggestions():
    query = request.args.get('q', '')
    if not query:
        return jsonify({'error': 'Query parameter missing'})

    results = []
    try:
        with WEBS() as webs:
            for result in webs.suggestions(query):
                results.append(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'results': results})

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'working'})


@app.route('/api/tts', methods=['POST'])
def ttd_endpoint():
    data = request.get_json()
    message = data.get('message')
    voice = data.get('voice', "Brian") # Default voice is "Brian"
    
    if not message:
        return jsonify({'error': 'Message parameter missing'}), 400
    
    audio_file = text_to_speech(message, voice=voice)
    return send_file(audio_file, mimetype='audio/mpeg')


@app.route('/api/image-generator', methods=['POST'])
def image_generator_endpoint():
    data = request.get_json()
    filename = data.get('filename', '')
    number = data.get('number', 1)
    prompt = data.get('prompt', 'RANDOM')
    prompt_size = data.get('prompt_size', 10)
    negative_prompt = data.get('negative_prompt', 'text')
    style = data.get('style', 'RANDOM')
    resolution = data.get('resolution', '512x768')
    guidance_scale = data.get('guidance_scale', 7)
    
    try:
        generate_images(filename=filename, number=number, prompt=prompt, prompt_size=prompt_size, negative_prompt=negative_prompt, style=style, resolution=resolution, guidance_scale=guidance_scale)
        return jsonify({'message': 'Images generated successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
@app.route('/api/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    image_path = data.get('image_path')
    text = data.get('text')
    decoding_strategy = data.get('decoding_strategy', "Greedy")
    temperature = data.get('temperature', 0.4)
    max_new_tokens = data.get('max_new_tokens', 512)
    repetition_penalty = data.get('repetition_penalty', 1.2)
    top_p = data.get('top_p', 0.8)
    
    if not image_path or not text:
        return jsonify({'error': 'Image path and text parameters are required'}), 400
    
    result = predict_with_model(image_path, text, decoding_strategy, temperature, max_new_tokens, repetition_penalty, top_p)
    return jsonify(result)

def generate_video_from_text(prompt, base="epiCRealism", motion="", step="4"):
    """
    Generates a video from the given text using the specified base model, motion, and step.

    Parameters:
    - prompt (str): The text to be converted into a video.
    - base (str): The base model to use for generating the video. Default is "epiCRealism".
    - motion (str): The motion to apply to the video. Default is an empty string, which means no motion.
    - step (str): The number of inference steps to use. Default is "4".

    Returns:
    - dict: A dictionary containing the video filepath and subtitles filepath (if any).
    """
    # Create a client instance
    client = Client("Gradio-Community/Animation_With_Sound")

    # Send a request to the API with the specified parameters
    result = client.predict(
        prompt=prompt,
        base=base,
        motion=motion,
        step=step,
        api_name="/generate_image"
    )

    # Return the result
    return result

@app.route('/generate_video', methods=['POST'])
def generate_video():
    # Extract parameters from the request 
    data = request.get_json()
    prompt = data.get('prompt', '')
    base = data.get('base', 'epiCRealism')

    step = data.get('step', '4')

    # Generate the video
    if base=="Realistic":
     result = generate_video_from_text(prompt,"epiCRealism",  step)
    if base=="Anime":
     result = generate_video_from_text(prompt,"ToonYou", step)

    # Return the result as a JSON response
    return jsonify(result)



@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if file:
        recognizer = sr.Recognizer()
        audio_data = sr.AudioFile(file)
        with audio_data as source:
            audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio, language='en-US') # Default to English
            return jsonify({"text": text})
        except sr.UnknownValueError:
            return jsonify({"error": "Google Speech Recognition could not understand audio"}), 400
        except sr.RequestError as e:
            return jsonify({"error": f"Could not request results from Google Speech Recognition service; {e}"}), 500

@app.route('/api/transcript', methods=['GET'])
def transcript_endpoint():
    video_id = request.args.get('video_id')
    if not video_id:
        return jsonify({'error': 'Video ID parameter is required'}), 400

    transcriber_instance = TranscriberYT()
    transcript = transcriber_instance.extract_transcript(video_id)
    return jsonify({'transcript': transcript})

def predict_with_model(image_url, text, decoding_strategy="Greedy", temperature=0.4, max_new_tokens=512, repetition_penalty=1.2, top_p=0.8):
    # Initialize the Client object with the model identifier
    client = Client("HuggingFaceM4/idefics-8b")

    # Call the predict method with the provided parameters
    result = client.predict(
        image=file(image_url),
        text=text,
        decoding_strategy=decoding_strategy,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        top_p=top_p,
        api_name="/model_inference"
    )

    # Return the result
    return result


def Cquery(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

@app.route('/caption', methods=['POST'])
def caption_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if file:
        filename = file.filename
        file.save(filename)
        output = Cquery(filename)
        return jsonify(output)




if __name__ == "__main__":
    app.run()
