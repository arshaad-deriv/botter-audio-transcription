# Set environment variable to prevent pygame from initializing GUI components
# This fixes the "setting the main menu on a non-main thread" error on macOS
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import streamlit as st

# Set page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="Audio Transcription App",
    page_icon="🎙️",
    layout="wide",
)

import tempfile
import time
import numpy as np
# Import sounddevice conditionally
import sys
import platform
import tempfile
import time
import base64
import queue
import threading
import io
import json
import math
import requests
import wave  # Add this import at the top with other imports
import re

# Detect if running in Streamlit Cloud
is_cloud_env = os.environ.get('STREAMLIT_RUNTIME_IS_STREAMLIT_CLOUD') == 'true' or 'HOSTNAME' in os.environ

# Initialize sounddevice as None
sd = None

# Try to import sounddevice only if not in cloud environment
if not is_cloud_env:
    try:
        import sounddevice as sd
        import soundfile as sf
        AUDIO_RECORDING_AVAILABLE = True
    except (ImportError, OSError) as e:
        # Store warning message for later display to avoid st commands before set_page_config
        audio_import_error = f"Audio recording disabled: {str(e)}"
        AUDIO_RECORDING_AVAILABLE = False
else:
    AUDIO_RECORDING_AVAILABLE = False
    # Store warning message for later display
    cloud_warning = "Audio recording is not available in cloud environments."

# Continue with other imports that should work anywhere
import openai
import anthropic
from pydub import AudioSegment
import base64
import websocket
import audioop
import struct
from moviepy.editor import VideoFileClip  # For MP4 to MP3 conversion

# Load environment variables (for local development with .env file)
# load_dotenv()

# Function to get API keys from Streamlit secrets or environment variables with fallback to user input
def get_api_key(key_name):
    """
    Get API key from Streamlit secrets or environment variables with fallback
    
    Args:
        key_name: Name of the API key to retrieve
        
    Returns:
        API key value or None if not found
    """
    # First try to get from Streamlit secrets
    if key_name in st.secrets:
        return st.secrets[key_name]
    
    # Then try environment variables
    return os.environ.get(key_name)

# ======================================
# FUNCTION DEFINITIONS SECTION
# ======================================

# Helper function to create a download link for text content
def get_text_download_link(text, filename, link_text):
    """Generate a link to download text content as a file without triggering a page rerun"""
    # Create a base64 encoded version of the text
    b64 = base64.b64encode(text.encode()).decode()
    
    # Create an HTML download link
    html_link = f'''
    <a href="data:text/plain;base64,{b64}" download="{filename}" 
       style="display: inline-block; padding: 0.375rem 0.75rem; 
              font-size: 1rem; font-weight: 400; line-height: 1.5; 
              text-align: center; text-decoration: none; vertical-align: middle; 
              cursor: pointer; border: 1px solid transparent; 
              border-radius: 0.25rem; background-color: #4CAF50; 
              color: white; margin: 5px 0px;">
        {link_text}
    </a>
    '''
    
    # Return the HTML to be displayed
    return st.markdown(html_link, unsafe_allow_html=True)

# Function to record audio from microphone - only available in local environments
def record_audio(duration):
    """Record audio from microphone for specified duration"""
    if not AUDIO_RECORDING_AVAILABLE or sd is None:
        st.error("Microphone recording is not available in this environment")
        return None
    
    sample_rate = 44100  # Sample rate in Hz
    
    # Create a temporary file to store the recording
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    temp_file_path = temp_file.name
    temp_file.close()
    
    # Record audio from microphone
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # Wait for recording to complete
    
    # Save the recording to the temporary file
    sf.write(temp_file_path, recording, sample_rate)
    
    return temp_file_path

# Function to split audio file into chunks of specified size
def split_audio_file(audio_file_path, max_chunk_size_mb=24.5):
    """
    Split a large audio file into chunks of specified size.
    
    Args:
        audio_file_path: Path to the audio file
        max_chunk_size_mb: Maximum size of each chunk in MB (default: 24.5MB for OpenAI API limit)
        
    Returns:
        List of paths to the chunk files
    """
    try:
        # Load audio file using pydub
        audio = AudioSegment.from_file(audio_file_path)
        
        # Get file extension
        file_extension = os.path.splitext(audio_file_path)[1].lower()
        
        # Calculate file size in bytes
        file_size = os.path.getsize(audio_file_path)
        file_size_mb = file_size / (1024 * 1024)
        
        # If file is smaller than the max chunk size, return the original file
        if file_size_mb <= max_chunk_size_mb:
            return [audio_file_path]
        
        # Calculate the number of chunks needed
        num_chunks = math.ceil(file_size_mb / max_chunk_size_mb)
        
        # Calculate chunk duration in milliseconds
        total_duration_ms = len(audio)
        chunk_duration_ms = total_duration_ms // num_chunks
        
        # Create chunks
        chunk_paths = []
        for i in range(num_chunks):
            # Calculate start and end time for this chunk
            start_time = i * chunk_duration_ms
            end_time = min((i + 1) * chunk_duration_ms, total_duration_ms)
            
            # Extract chunk
            chunk = audio[start_time:end_time]
            
            # Create temporary file for the chunk - always use mp3 for better compatibility
            temp_chunk_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            chunk_path = temp_chunk_file.name
            temp_chunk_file.close()
            
            # Export chunk to file - always use mp3 format for better compatibility
            chunk.export(chunk_path, format='mp3')
            
            # Add to list of chunk paths
            chunk_paths.append(chunk_path)
        
        return chunk_paths
    
    except Exception as e:
        st.error(f"Error splitting audio file: {str(e)}")
        print(f"Audio splitting error: {str(e)}")
        return [audio_file_path]  # Return original file if splitting fails

# Function to transcribe audio using OpenAI API
def transcribe_audio(audio_file_path, model="whisper-1", translate=False):
    """Transcribe audio file using OpenAI's API"""
    try:
        # Get API key from secrets or environment variables
        api_key = get_api_key("OPENAI_API_KEY")
        
        # Create a client instance
        client = openai.OpenAI(api_key=api_key)
        
        # Check if file needs to be split into chunks
        file_size = os.path.getsize(audio_file_path) / (1024 * 1024)  # Size in MB
        
        if file_size > 24.5:
            st.info(f"File size ({file_size:.2f} MB) exceeds OpenAI's limit. Splitting into chunks...")
            
            # Split audio file into chunks
            chunk_paths = split_audio_file(audio_file_path)
            
            # Transcribe each chunk
            combined_transcription = ""
            progress_bar = st.progress(0)
            
            for i, chunk_path in enumerate(chunk_paths):
                chunk_status = st.empty()
                chunk_status.info(f"Transcribing chunk {i+1} of {len(chunk_paths)}...")
                
                with open(chunk_path, "rb") as audio_file:
                    if translate:
                        response = client.audio.translations.create(
                            model=model,
                            file=audio_file,
                            response_format="text"
                        )
                    else:
                        response = client.audio.transcriptions.create(
                            model=model,
                            file=audio_file,
                            response_format="text"
                        )
                
                # Extract text from response
                if isinstance(response, str):
                    chunk_text = response
                elif isinstance(response, dict) and "text" in response:
                    chunk_text = response["text"]
                elif hasattr(response, "text"):
                    chunk_text = response.text
                else:
                    chunk_text = str(response)
                
                # Add to combined transcription with a space between chunks if needed
                if combined_transcription and not combined_transcription.endswith(" ") and not chunk_text.startswith(" "):
                    combined_transcription += " "
                combined_transcription += chunk_text
                
                # Update progress
                progress_bar.progress((i + 1) / len(chunk_paths))
                
                # Clean up chunk file if it's not the original file
                if chunk_path != audio_file_path:
                    os.unlink(chunk_path)
                
                chunk_status.empty()
            
            st.success("All chunks transcribed successfully!")
            return combined_transcription.strip()
            
        else:
            # Original implementation for files under the size limit
            with open(audio_file_path, "rb") as audio_file:
                if translate:
                    response = client.audio.translations.create(
                        model=model,
                        file=audio_file,
                        response_format="text"
                    )
                else:
                    response = client.audio.transcriptions.create(
                        model=model,
                        file=audio_file,
                        response_format="text"
                    )
            
            # In the new API, the response is the text directly when using response_format="text"
            if isinstance(response, str):
                return response
            # Fallback for other response formats
            elif isinstance(response, dict) and "text" in response:
                return response["text"]
            elif hasattr(response, "text"):
                return response.text
            else:
                return str(response)
    
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        print(f"API error: {str(e)}")
        return None

# Function to extract audio from video files
def extract_audio_from_video(video_file_path):
    """Extract audio from MP4 video and save as MP3"""
    try:
        # Create temporary file for the extracted audio
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_audio_path = temp_audio_file.name
        temp_audio_file.close()
        
        # Load the video and extract audio
        video = VideoFileClip(video_file_path)
        audio = video.audio
        
        # Get video duration
        duration_seconds = video.duration
        
        # Format duration as HH:MM:SS
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        seconds = int(duration_seconds % 60)
        duration_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        # Write audio to MP3 file with additional ffmpeg parameters for better compatibility
        audio.write_audiofile(
            temp_audio_path,
            codec='libmp3lame',  # Explicitly use libmp3lame codec
            bitrate='192k',
            ffmpeg_params=['-ac', '1'],  # Convert to mono
            logger=None  # Suppress ffmpeg output
        )
        
        # Get file size in MB
        file_size_mb = os.path.getsize(temp_audio_path) / (1024 * 1024)
        
        # Close the video to free resources
        video.close()
        audio.close()
        
        return temp_audio_path, file_size_mb, duration_formatted
    
    except Exception as e:
        st.error(f"Error extracting audio: {str(e)}")
        print(f"Video processing error: {str(e)}")
        return None, 0, "00:00:00"

# Real-time transcription class
class RealtimeTranscription:
    def __init__(self, api_key):
        self.api_key = api_key
        self.ws = None
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.is_connected = False
        self.transcript = ""
        self.error_message = None
        
    def get_error(self):
        return self.error_message
    
    def on_message(self, ws, message):
        response = json.loads(message)
        event_type = response.get("type")
        
        if event_type == "error":
            self.error_message = response.get("error")
            print(f"WebSocket error: {response.get('error')}")
        
        elif event_type == "conversation.item.input_audio_transcription.delta":
            delta = response.get("delta", "")
            self.transcript += delta
        
        elif event_type == "conversation.item.input_audio_transcription.completed":
            self.transcript = response.get("transcript", self.transcript)
    
    def on_error(self, ws, error):
        self.error_message = str(error)
        print(f"WebSocket error: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        print("WebSocket closed")
        self.is_connected = False
    
    def on_open(self, ws):
        print("WebSocket connection established")
        self.is_connected = True
        
        # Start audio capture and streaming thread
        audio_thread = threading.Thread(target=self.capture_and_stream_audio)
        audio_thread.daemon = True
        audio_thread.start()
    
    def capture_and_stream_audio(self):
        if not AUDIO_RECORDING_AVAILABLE or sd is None:
            self.error_message = "Microphone recording is not available in this environment"
            self.stop()
            return
            
        # Audio parameters
        sample_rate = 16000
        chunk_size = 1024
        
        try:
            # Open audio stream
            stream = sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                callback=self.audio_callback
            )
            stream.start()
            
            # Keep streaming until stop event is set
            while not self.stop_event.is_set():
                if not self.audio_queue.empty() and self.ws and self.is_connected:
                    audio_data = self.audio_queue.get()
                    
                    # Convert audio to base64 and send to WebSocket
                    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                    payload = json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": audio_base64
                    })
                    self.ws.send(payload)
                
                time.sleep(0.01)  # Small pause to reduce CPU usage
            
            stream.stop()
            stream.close()
            
        except Exception as e:
            self.error_message = str(e)
            print(f"Audio streaming error: {e}")
    
    def audio_callback(self, indata, frames, time_info, status):
        # Convert float32 audio data to int16
        audio_data = (indata * 32767).astype(np.int16).tobytes()
        
        # Add to queue for sending
        self.audio_queue.put(audio_data)
    
    def start(self):
        # For cloud environments, don't attempt to use microphone
        if not AUDIO_RECORDING_AVAILABLE and 'is_cloud_env' in globals() and is_cloud_env:
            self.error_message = "Real-time transcription requires microphone access, which is not available in Streamlit Cloud environment."
            return False

        try:
            # First create a transcription session via REST API
            url = "https://api.openai.com/v1/realtime/transcription_sessions"
            payload = {
                "input_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "gpt-4o-mini-transcribe",  # Using mini model here
                    "language": "en",
                    "prompt": "Transcribe the incoming audio in real time."
                },
                "turn_detection": {"type": "server_vad", "silence_duration_ms": 1000}
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "OpenAI-Beta": "assistants=v2"
            }
            
            response = requests.post(url, json=payload, headers=headers)
            if response.status_code != 200:
                self.error_message = f"Failed to create transcription session: {response.status_code} {response.text}"
                return False
                
            data = response.json()
            ephemeral_token = data["client_secret"]["value"]
            
            # Now connect to WebSocket with ephemeral token
            headers = {
                "Authorization": f"Bearer {ephemeral_token}",
                "OpenAI-Beta": "realtime=v1"
            }
            
            # Initialize WebSocket connection
            self.ws = websocket.WebSocketApp(
                "wss://api.openai.com/v1/realtime",
                header=headers,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )
            
            # Start WebSocket connection in a background thread
            self.stop_event.clear()
            ws_thread = threading.Thread(target=self.ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            return True
            
        except Exception as e:
            self.error_message = str(e)
            print(f"WebSocket connection error: {e}")
            return False
    
    def stop(self):
        self.stop_event.set()
        if self.ws:
            self.ws.close()
    
    def get_transcript(self):
        return self.transcript

# New class for handling chunked audio processing
class ChunkedAudioProcessor:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)
        self.audio_buffer = b''
        self.chunk_duration_ms = 600  # 600ms chunks
        self.sample_rate = 16000
        self.channels = 1
        self.transcript = ""
        self.translation = ""
        self.is_processing = False
        self.processing_lock = threading.Lock()
        self.last_chunk_time = time.time()
        self.language = "Unknown"
        
    def add_audio_chunk(self, audio_data):
        """Add a chunk of audio data to the buffer"""
        self.audio_buffer += audio_data
        current_time = time.time()
        
        # Process chunks if we have enough data and not currently processing
        if len(self.audio_buffer) >= (self.sample_rate * self.channels * 2 * (self.chunk_duration_ms / 1000)) and not self.is_processing:
            self.is_processing = True
            chunk_thread = threading.Thread(target=self.process_audio_chunk)
            chunk_thread.daemon = True
            chunk_thread.start()
            self.last_chunk_time = current_time
    
    def process_audio_chunk(self):
        """Process the current audio buffer into transcription and translation"""
        try:
            # Create a copy of the buffer and clear the original
            with self.processing_lock:
                audio_chunk = self.audio_buffer
                self.audio_buffer = b''
            
            # Save chunk to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_file_path = temp_file.name
            temp_file.close()
            
            # Convert raw audio data to WAV file
            with wave.open(temp_file_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_chunk)
            
            # Transcribe the audio chunk
            with open(temp_file_path, 'rb') as audio_file:
                response = self.client.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe",
                    file=audio_file,
                    response_format="text"
                )
            
            # Append new transcription
            chunk_text = response
            if chunk_text and chunk_text.strip():
                self.transcript += " " + chunk_text
                
                # Detect language if it's still unknown
                if self.language == "Unknown" and len(self.transcript) > 20:
                    self.detect_language()
                
                # Translate if not English
                if self.language not in ["English", "Unknown"]:
                    self.translate_chunk(chunk_text)
            
            # Clean up
            os.unlink(temp_file_path)
            
        except Exception as e:
            print(f"Error processing audio chunk: {e}")
        finally:
            self.is_processing = False
    
    def detect_language(self):
        """Detect the language of the transcript"""
        try:
            detect_prompt = f"""Determine the language of this text. Just respond with the language name:
            
            Text: {self.transcript[:100]}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You detect languages. Reply with just the language name."},
                    {"role": "user", "content": detect_prompt}
                ],
                temperature=0.2,
                max_tokens=10
            )
            
            self.language = response.choices[0].message.content.strip()
            print(f"Detected language: {self.language}")
            
        except Exception as e:
            print(f"Error detecting language: {e}")
    
    def translate_chunk(self, text):
        """Translate the latest chunk to English"""
        try:
            translate_prompt = f"""Translate this {self.language} text to English:
            
            {text}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": f"You are a {self.language} to English translator. Translate directly without explanations."},
                    {"role": "user", "content": translate_prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            
            translated_text = response.choices[0].message.content.strip()
            if translated_text:
                self.translation += " " + translated_text
            
        except Exception as e:
            print(f"Error translating text: {e}")
    
    def get_transcript(self):
        """Get the current transcript"""
        return self.transcript.strip()
    
    def get_translation(self):
        """Get the current translation"""
        return self.translation.strip()
    
    def get_language(self):
        """Get the detected language"""
        return self.language

# Function to detect language and translate non-English transcriptions to English
def detect_and_translate(text):
    """
    Detect if the text is in English and translate it to English if it's not.
    Uses GPT-4.1-mini for language detection and translation.
    
    Args:
        text: The transcription text to detect language and possibly translate
        
    Returns:
        Tuple containing (is_english, translated_text, detected_language)
    """
    try:
        # Get API key
        api_key = get_api_key("OPENAI_API_KEY")
        if not api_key:
            return True, text, "Unknown"  # Assume English if no API key
        
        # Create a client instance
        client = openai.OpenAI(api_key=api_key)
        
        # First, detect the language
        detect_prompt = f"""Analyze the following text and determine the language:

Text: {text[:500]}...  # Only using first 500 chars for detection

Your response should be in the format:
Language: [language name]
Is English (yes/no): [yes or no]
Confidence (1-10): [1-10 scale where 10 is highest]

Keep your response brief and to the point."""
        
        detect_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a language detection specialist. Analyze the text and identify its language. Be brief and precise."},
                {"role": "user", "content": detect_prompt}
            ],
            temperature=0.3
        )
        
        detection_result = detect_response.choices[0].message.content
        print(f"Language detection result: {detection_result}")
        
        # Parse detection result - changed to be more lenient
        is_english = "yes" in detection_result.lower() and "is english" in detection_result.lower()
        
        # If confidence level is mentioned and is low for English, don't treat as English
        confidence_level = 10  # Default high confidence
        for line in detection_result.split('\n'):
            if line.lower().startswith("confidence"):
                try:
                    # Extract the confidence value (1-10)
                    confidence_parts = line.split(':')
                    if len(confidence_parts) > 1:
                        confidence_str = confidence_parts[1].strip()
                        # Extract just the number
                        confidence_str = ''.join(c for c in confidence_str if c.isdigit())
                        if confidence_str:
                            confidence_level = int(confidence_str)
                except ValueError:
                    # If parsing fails, keep default
                    pass
        
        # If it's detected as English but with low confidence (< 7), don't treat as English
        if is_english and confidence_level < 7:
            is_english = False
            print(f"Detected as English but with low confidence ({confidence_level}/10), treating as non-English")
        
        # Extract language name
        language_name = "Unknown"
        for line in detection_result.split('\n'):
            if line.lower().startswith("language:"):
                language_name = line.split(':', 1)[1].strip()
                break
        
        # For testing - always show the detected language and confidence in console
        print(f"Detected language: {language_name}, Is English: {is_english}, Confidence: {confidence_level}/10")
        
        # If not English, translate
        if not is_english:
            translate_prompt = f"""Translate the following text from {language_name} to English:

{text}

Provide only the translated text without any additional comments or notes."""
            
            translate_response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are a professional translator. Translate the given text to English while preserving the meaning, tone, and style of the original."},
                    {"role": "user", "content": translate_prompt}
                ],
                temperature=0.3
            )
            
            translated_text = translate_response.choices[0].message.content
            return False, translated_text, language_name
        
        # If English, return the original text
        return True, text, language_name
        
    except Exception as e:
        st.error(f"Error detecting language or translating: {str(e)}")
        print(f"Translation API error: {str(e)}")
        return True, text, "Unknown"  # Assume English on error

# Function to generate a summary using GPT-4
def generate_summary(text, summary_type):
    """Generate a summary of the transcription using OpenAI's GPT models"""
    try:
        # Get API key from secrets or environment variables
        api_key = get_api_key("OPENAI_API_KEY")
        if not api_key:
            st.warning("OpenAI API key is required for generating summaries. Please enter it in the sidebar.")
            return None
            
        # Create a client instance with API key
        client = openai.OpenAI(api_key=api_key)
        
        # Choose model and prompt based on Whisper model availability and preferences
        model = "gpt-4.1"  # Fallback to GPT-4o Mini as it's sufficient for summary
        
        if summary_type == "meeting":
            system_prompt = """You are a professional meeting summarizer specialized in creating structured, concise summaries of conversations and meetings.
            Your summaries should highlight the key points, decisions made, action items, and next steps discussed.
            Present your summary in a structured format with the following sections:
            
            - Overall Summary: A brief 2-3 sentence overview of what the meeting was about
            - Completed: Items/tasks reported as completed
            - Ongoing: Tasks/projects currently in progress
            - Blockers: Any obstacles or issues mentioned
            - Ideas Discussed: Key concepts or suggestions that were brought up
            - To Do: Specific action items and who they're assigned to
            - Action Points We Need to Start as a Team
            
            Make sure to format each section with appropriate bullet points and include names of people mentioned for action items when available. If a particular section has no content, indicate "None mentioned" under that heading."""
            
            user_prompt = f"Please summarize the following meeting transcript in a structured format highlighting key points, decisions and action items:\n\n{text}"
            
        else:  # general summary
            system_prompt = """You are a professional content summarizer capable of condensing spoken text into clear, concise summaries.
            Focus on the main ideas, key points, and overall message while removing filler words, repetitions, and unimportant details.
            Maintain a professional but conversational tone in your summary, and organize the information logically."""
            
            user_prompt = f"Please provide a concise general summary of the following transcription:\n\n{text}"
        
        # Make API call using the initialized client
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        
        # Extract and return the summary
        return response.choices[0].message.content
    
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        print(f"GPT API error: {str(e)}")
        return None

# Function to generate a summary using Claude API
def generate_claude_summary(text, summary_type):
    """Generate a summary using Claude 3.7 API"""
    try:
        # Check if Claude API key is available
        claude_key = get_api_key("CLAUDE_API_KEY")
        if not claude_key:
            st.warning("Claude API key is required for summaries. Please enter it in the sidebar.")
            return None
        
        # Initialize client with Claude API key
        claude_client = anthropic.Anthropic(api_key=claude_key)
        
        if summary_type == "meeting":
            system_prompt = """You are an experienced journalist at The New York Times."""
            
            user_prompt = f"""Analyze the following meeting transcript and rewrite it into a series of news headlines and corresponding subheadlines (decks) that reflect the publication's editorial standards while highlighting key aspects of the meeting.
<meeting_transcript>
{text}
</meeting_transcript>
After carefully reading the transcript, create headlines and subheadlines for each of the following elements:

1. A masthead with the following
Title: THE MEETING NEWS
Subtitle: A headline for the meeting
2. Main Story: Capture the meeting's purpose and outcome.
3. The Good: Highlight positive accomplishments, KPIs, or metrics worth escalating.
4. The Bad: Address neutral or somewhat negative issues, KPIs, or metrics worth escalating.
5. The Ugly: Focus on serious or high-risk issues, including KPIs, metrics, blockers, or other issues worth escalating.
6. Issues Discussed: Cover each significant topic discussed in the meeting.
7. Key Decisions: Highlight explicit decisions reached during the meeting.
8. Assigned Actions: Detail specific tasks assigned, including who they were assigned to and timelines if mentioned.
9. Questions & Issues Raised:
a. Key questions raised and clearly resolved
b. Key questions raised but left unresolved or open
c. Key questions that should have been raised but were not
10. Blockers & Challenges: Highlight obstacles, blockers, or hard challenges raised.
11. Follow-up Discussions & Check-ins: Mention any follow-up discussions or check-ins, including dates and objectives.
12. Key People Referenced: List individuals mentioned (including non-attendees) and their roles if stated. Only include people explicitly mentioned in the meeting. Do not make up implied names for those not mentioned.
For each element:
13. Extract Key Stories: Identify the most newsworthy elements within the text.
14. Craft Headlines: For each key story, write a concise, informative headline that captures the essence of the news. Use clear and formal language, adhering to The New York Times' style.
15. Write Subheadlines (Decks): Below each headline, provide a subheadline that offers additional context or details, maintaining the same journalistic tone.
Ensure your writing maintains journalistic integrity by prioritizing accuracy, neutrality, and clarity. Avoid sensationalism and maintain the objective tone characteristic of The New York Times.
Format your output as a list of headline and subheadline pairs suitable for publication on The New York Times' homepage.
Your final answer should only include the formatted list of headlines and subheadlines, organized under the categories mentioned above. Do not include any additional commentary or explanations outside of the requested format."""
            
        else:  # general summary
            system_prompt = """You are a professional content summarizer capable of condensing spoken text into clear, concise summaries.
            Focus on the main ideas, key points, and overall message while removing filler words, repetitions, and unimportant details.
            Maintain a professional but conversational tone in your summary, and organize the information logically."""
            
            user_prompt = f"Please provide a concise general summary of the following transcription:\n\n{text}"
        
        # Call Claude API
        response = claude_client.messages.create(
            model="claude-3-7-sonnet-latest",
            max_tokens=20000,  # Increased from 4000 to 50000
            temperature=0.3,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        
        # Extract the summary from the response
        summary = response.content[0].text
        
        return summary
        
    except Exception as e:
        st.error(f"Error generating Claude summary: {str(e)}")
        print(f"Claude API error: {str(e)}")
        return None

# New function to format the New York Times style headlines
def format_nyt_summary(summary_text):
    # Format the NYT style headline summary
    formatted_text = summary_text
    
    # Add category styling for each section
    section_titles = {
        "THE MEETING NEWS": "<div class='nyt-masthead'>📰 THE MEETING NEWS</div>",
        "Main Story:": "<div class='nyt-section'>📑 MAIN STORY</div>",
        "The Good:": "<div class='nyt-section'>✅ THE GOOD</div>",
        "The Bad:": "<div class='nyt-section'>⚠️ THE BAD</div>",
        "The Ugly:": "<div class='nyt-section'>🚫 THE UGLY</div>",
        "Issues Discussed:": "<div class='nyt-section'>💬 ISSUES DISCUSSED</div>",
        "Key Decisions:": "<div class='nyt-section'>🔑 KEY DECISIONS</div>",
        "Assigned Actions:": "<div class='nyt-section'>📋 ASSIGNED ACTIONS</div>",
        "Questions & Issues Raised:": "<div class='nyt-section'>❓ QUESTIONS & ISSUES RAISED</div>",
        "Blockers & Challenges:": "<div class='nyt-section'>🚧 BLOCKERS & CHALLENGES</div>",
        "Follow-up Discussions & Check-ins:": "<div class='nyt-section'>📅 FOLLOW-UP DISCUSSIONS</div>",
        "Key People Referenced:": "<div class='nyt-section'>👥 KEY PEOPLE REFERENCED</div>"
    }
    
    for original, styled in section_titles.items():
        formatted_text = formatted_text.replace(original, styled)
    
    # Format headlines and subheadlines
    lines = formatted_text.split("\n")
    result_lines = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            result_lines.append(line)
            continue
            
        # Check if this line is a headline
        is_headline = False
        
        # Headlines typically start with numbers or letters followed by period/colon
        if re.match(r'^[a-zA-Z0-9][a-zA-Z0-9]?[\.\:)]', stripped) or stripped.startswith("- "):
            is_headline = True
            
        # Format headline
        if is_headline:
            # Remove the bullet or numbering
            headline_text = re.sub(r'^[a-zA-Z0-9][a-zA-Z0-9]?[\.\:)] ', '', stripped)
            headline_text = re.sub(r'^- ', '', headline_text)
            
            # Wrap headline in styled div
            result_lines.append(f"<div class='nyt-headline'>{headline_text}</div>")
            
            # Check if next line is a subheadline (non-empty and not a new headline)
            if i+1 < len(lines) and lines[i+1].strip() and not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9]?[\.\:)]', lines[i+1].strip()) and not lines[i+1].strip().startswith("- "):
                # It's a subheadline
                subheadline = lines[i+1].strip()
                result_lines.append(f"<div class='nyt-subheadline'>{subheadline}</div>")
                # Skip the next line as we've already processed it
                lines[i+1] = ""
            
        # If not a headline and not already processed as a subheadline
        elif stripped:
            result_lines.append(line)
    
    formatted_text = "\n".join(result_lines)
    
    # Add custom CSS for newspaper styling
    st.markdown("""
    <style>
    .nyt-masthead {
        font-family: 'Georgia', serif;
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        margin: 20px 0 5px 0;
        border-bottom: 2px solid #000;
        padding-bottom: 5px;
    }
    .nyt-section {
        font-family: 'Georgia', serif;
        font-size: 1.5em;
        font-weight: bold;
        margin: 25px 0 15px 0;
        border-bottom: 1px solid #ddd;
        padding-bottom: 5px;
    }
    .nyt-headline {
        font-family: 'Georgia', serif;
        font-size: 1.2em;
        font-weight: bold;
        margin: 15px 0 5px 0;
    }
    .nyt-subheadline {
        font-family: 'Georgia', serif;
        font-style: italic;
        color: #555;
        margin: 0 0 15px 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    return formatted_text

# Function to format the meeting summary with category styling and proper list handling
def format_meeting_summary(summary_text):
    # Add category styling with icons
    formatted_text = summary_text
    
    # Highlight Overall Summary section in yellow with black text
    formatted_text = formatted_text.replace("Overall Summary:", "<span class='category-overall'>📝 OVERALL SUMMARY</span>")
    
    # Add category styling with icons for other sections
    formatted_text = formatted_text.replace("- Completed:", "<span class='category-completed'>✓ COMPLETED</span>")
    formatted_text = formatted_text.replace("- Ongoing:", "<span class='category-ongoing'>⟳ ONGOING</span>")
    formatted_text = formatted_text.replace("- Blockers:", "<span class='category-blockers'>⚠ BLOCKERS</span>")
    formatted_text = formatted_text.replace("- Ideas Discussed:", "<span class='category-ideas'>💡 IDEAS</span>")
    formatted_text = formatted_text.replace("- To Do:", "<span class='category-todo'>📋 TODO</span>")
    formatted_text = formatted_text.replace("Action Points We Need to Start as a Team:", "<span class='category-action'>🚀 ACTION POINTS</span>")
    
    # Further formatting to make Claude's output more like OpenAI's
    # Convert inline category headings to standalone headings with line breaks
    for category in ["COMPLETED", "ONGOING", "BLOCKERS", "IDEAS", "TODO", "ACTION POINTS"]:
        formatted_text = formatted_text.replace(f"<span class='category-{category.lower()}'>{category}</span>", 
                                             f"<div class='category-heading'><span class='category-{category.lower()}'>{category}</span></div>")
    
    # Special case for Overall Summary
    formatted_text = formatted_text.replace("<span class='category-overall'>📝 OVERALL SUMMARY</span>", 
                                         "<div class='category-heading'><span class='category-overall'>📝 OVERALL SUMMARY</span></div>")
    
    # Handle "None mentioned" consistently - this fixes the weird formatting
    formatted_text = formatted_text.replace("None mentioned.", "<ul><li>None mentioned</li></ul>")
    formatted_text = formatted_text.replace("None mentioned", "<ul><li>None mentioned</li></ul>")
    
    # Convert bullet points in Claude format to proper HTML lists
    lines = formatted_text.split("\n")
    result_lines = []
    in_list = False
    current_list_items = []
    
    for line in lines:
        stripped = line.strip()
        # Handle bullet points at the start of lines
        if stripped.startswith("•") or stripped.startswith("* ") or stripped.startswith("- "):
            if not in_list:
                in_list = True
            # Extract the text after the bullet point
            if stripped.startswith("•"):
                item_text = stripped[1:].strip()
            elif stripped.startswith("* "):
                item_text = stripped[2:].strip()
            elif stripped.startswith("- "):
                item_text = stripped[2:].strip()
            
            if item_text:  # Only add non-empty items
                current_list_items.append(item_text)
        else:
            # If we were in a list and now we're not, close the list
            if in_list:
                if current_list_items:
                    result_lines.append("<ul>")
                    for item in current_list_items:
                        result_lines.append(f"<li>{item}</li>")
                    result_lines.append("</ul>")
                    current_list_items = []
                in_list = False
            result_lines.append(line)
    
    # Don't forget to add any remaining list items
    if in_list and current_list_items:
        result_lines.append("<ul>")
        for item in current_list_items:
            result_lines.append(f"<li>{item}</li>")
        result_lines.append("</ul>")
    
    formatted_text = "\n".join(result_lines)
    
    # Fix any leftover bullet characters that might have been missed
    formatted_text = formatted_text.replace("• ", "")
    formatted_text = formatted_text.replace("•", "")
    
    return formatted_text

# Function to estimate token count for text
def estimate_token_count(text):
    """Estimate the number of tokens in a given text.
    
    This is a simple estimation based on the average ratio of tokens to characters.
    For English text, a common rule of thumb is ~4 characters per token.
    """
    # Simple estimation: ~4 characters per token for English text
    return len(text) // 4

# Function for handling large transcripts for Claude API
def prepare_transcript_for_claude(text, max_tokens=150000):
    """Prepare transcript for Claude API by truncating if it exceeds token limits.
    
    Args:
        text: The transcript text
        max_tokens: Maximum allowed tokens (default: 150000 to leave room for system prompt)
        
    Returns:
        Processed transcript that fits within token limits
    """
    estimated_tokens = estimate_token_count(text)
    
    if estimated_tokens <= max_tokens:
        return text
    
    # Calculate what percentage of the text we can keep
    keep_ratio = max_tokens / estimated_tokens
    
    # Truncate the text but preserve the beginning and end
    # Keep 70% from beginning and 30% from end to maintain context
    beginning_ratio = 0.7
    chars_to_keep = int(len(text) * keep_ratio)
    chars_from_beginning = int(chars_to_keep * beginning_ratio)
    chars_from_end = chars_to_keep - chars_from_beginning
    
    beginning = text[:chars_from_beginning]
    end = text[-chars_from_end:] if chars_from_end > 0 else ""
    
    # Add a note about truncation
    truncation_note = "\n\n[NOTE: This transcript has been truncated due to length. Some middle content has been removed to fit within API limits.]\n\n"
    
    return beginning + truncation_note + end

# Function for generating meeting evaluations with Claude 3.7
def generate_claude_meeting_evaluation(text):
    try:
        # Check if Claude API key is available
        claude_key = get_api_key("CLAUDE_API_KEY")
        if not claude_key:
            st.warning("Claude API key is required for Meeting Evaluation. Please enter it in the sidebar.")
            return None
        
        # Initialize client with Claude API key
        claude_client = anthropic.Anthropic(api_key=claude_key)
        
        # Prepare the transcript by truncating if needed
        estimated_tokens = estimate_token_count(text)
        if estimated_tokens > 150000:
            st.warning(f"Transcript is very large (approximately {estimated_tokens} tokens). Truncating to fit within Claude API limits.")
            prepared_text = prepare_transcript_for_claude(text)
        else:
            prepared_text = text
        
        # Use the new prompt provided by the user
        prompt = f"""You are a senior Project Manager tasked with evaluating a meeting transcript. Your goal is to assess the effectiveness of the discussion, identify areas for improvement, and provide constructive feedback based on project management best practices.

Here is the transcript of the meeting you need to analyze:

<meeting_transcript>
{prepared_text}
</meeting_transcript>

Please follow these steps to evaluate the meeting:

1. Carefully read through the meeting transcript.

2. Analyze the transcript, focusing on the following aspects:
   a) Meeting efficiency and productivity
   b) Clarity of communication and objectives
   c) Participant engagement and contribution
   d) Decision-making processes
   e) Action item assignment and follow-up
   f) Time management
   g) Overall meeting structure and flow

3. Inside your thinking block, wrap your analysis in <meeting_evaluation> tags. For each aspect:
   - Quote relevant parts of the transcript that support your observations.
   - Rate the aspect on a scale of 1-10, with 10 being excellent and 1 being poor.
   - List positive observations and areas for improvement separately.
   Pay special attention to issues related to tone, clarity, and team collaboration.

4. Based on your analysis, create a comprehensive evaluation of the meeting in the form of descriptive bullet points. Each bullet point should be 2-3 sentences long and focus on a specific aspect of the meeting. Include both positive observations and areas for improvement.

5. Assign an overall rating for the meeting discussion on a scale of 1 to 10, with 10 being excellent and 1 being poor. Justify your rating based on your observations.

6. Identify specific action points to address issues found in the meeting, particularly focusing on tone, clarity, and team collaboration.

7. Suggest improvements for future meetings to create a cycle of continuous improvement.

Your final output should be structured as follows:

<evaluation>
• [Bullet point 1]
• [Bullet point 2]
• [Bullet point 3]
...
• [Bullet point n]

Overall Rating: [X/10]
Justification: [2-3 sentences explaining the rating]

Action Points:
1. [Action point 1]
2. [Action point 2]
3. [Action point 3]

Suggestions for Future Meetings:
1. [Suggestion 1]
2. [Suggestion 2]
3. [Suggestion 3]
</evaluation>

Ensure that your evaluation is thorough, constructive, and provides actionable insights for improving future meetings. Focus on providing valuable feedback that can help enhance project management practices and team collaboration.

Your final output should consist only of the <evaluation> section and should not duplicate or rehash any of the work you did in the thinking block."""
        
        print("Generating meeting evaluation with Claude API...")
        
        # Call Claude API with retry on rate limit errors
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Call Claude API with a 60-second timeout
                response = claude_client.messages.create(
                    model="claude-3-7-sonnet-latest",
                    max_tokens=8000,
                    temperature=0.3,
                    system="You are a senior Project Manager with expertise in evaluating meeting effectiveness and providing constructive feedback.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    timeout=60
                )
                
                # Extract the evaluation from the response
                evaluation = response.content[0].text
                print("Claude meeting evaluation generated successfully!")
                return evaluation
                
            except Exception as e:
                error_message = str(e)
                print(f"Claude API error (attempt {attempt+1}/{max_retries}): {error_message}")
                
                # Check if it's a rate limit error
                if "rate_limit" in error_message.lower() and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # Exponential backoff: 5, 10, 15 seconds
                    print(f"Rate limit exceeded. Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                # If it's the final attempt or not a rate limit error, propagate the exception
                elif attempt == max_retries - 1:
                    st.error(f"Failed to generate meeting evaluation after {max_retries} attempts: {error_message}")
                    return None
                else:
                    st.error(f"Error generating Claude meeting evaluation: {error_message}")
                    return None
        
    except Exception as e:
        st.error(f"Error generating Claude meeting evaluation: {str(e)}")
        print(f"Claude API error: {str(e)}")
        return None

# Function to format the meeting evaluation with styles
def format_meeting_evaluation(evaluation_text):
    # Add styling to evaluation
    formatted_text = evaluation_text
    
    # Style bullet points for better readability
    lines = formatted_text.split("\n")
    result_lines = []
    in_list = False
    current_list_items = []
    
    for line in lines:
        stripped = line.strip()
        # Handle bullet points at the start of lines
        if stripped.startswith("•") or stripped.startswith("* ") or stripped.startswith("- "):
            if not in_list:
                in_list = True
            # Extract the text after the bullet point
            if stripped.startswith("•"):
                item_text = stripped[1:].strip()
            elif stripped.startswith("* "):
                item_text = stripped[2:].strip()
            elif stripped.startswith("- "):
                item_text = stripped[2:].strip()
            
            if item_text:  # Only add non-empty items
                current_list_items.append(item_text)
        # Detect "Overall Rating:" line
        elif "Overall Rating:" in stripped:
            # If we were in a list, close it
            if in_list:
                if current_list_items:
                    result_lines.append("<ul class='evaluation-list'>")
                    for item in current_list_items:
                        result_lines.append(f"<li>{item}</li>")
                    result_lines.append("</ul>")
                    current_list_items = []
                in_list = False
            # Add styled rating
            result_lines.append(f"<div class='evaluation-rating'>{stripped}</div>")
        # Detect "Justification:" line
        elif "Justification:" in stripped:
            result_lines.append(f"<div class='evaluation-justification'>{stripped}</div>")
        else:
            # If we were in a list and now we're not, close the list
            if in_list:
                if current_list_items:
                    result_lines.append("<ul class='evaluation-list'>")
                    for item in current_list_items:
                        result_lines.append(f"<li>{item}</li>")
                    result_lines.append("</ul>")
                    current_list_items = []
                in_list = False
            result_lines.append(line)
    
    # Don't forget to add any remaining list items
    if in_list and current_list_items:
        result_lines.append("<ul class='evaluation-list'>")
        for item in current_list_items:
            result_lines.append(f"<li>{item}</li>")
        result_lines.append("</ul>")
    
    formatted_text = "\n".join(result_lines)
    
    return formatted_text

# Function to update evaluation state without page rerun
def evaluation_callback():
    st.session_state.evaluation_clicked = True

# ======================================
# BEGIN STREAMLIT UI SECTION
# ======================================

# Display any warnings that were stored earlier
if not is_cloud_env and not AUDIO_RECORDING_AVAILABLE and 'audio_import_error' in locals():
    st.sidebar.warning(audio_import_error)
elif is_cloud_env:
    st.sidebar.warning(cloud_warning)

# Custom CSS (the duplicate set_page_config call was removed)
st.markdown("""
<style>
.main {
    padding: 2rem;
}
/* Removing the custom sidebar positioning that was causing issues */
/* div[data-testid="stSidebarContent"] {
    background-color: #1E1E1E;
    min-width: 300px !important;
    max-width: 300px !important;
    position: fixed !important;
    left: 0;
    top: 0;
    height: 100vh !important;
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.stApp {
    margin-left: 300px !important;
    max-width: calc(100% - 300px) !important;
} */
.header {
    text-align: center;
    margin-bottom: 2rem;
}
.transcription-container {
    background-color: #2E7D32;  /* Green background */
    border-radius: 10px;
    padding: 20px;
    margin-top: 20px;
    color: #FFFFFF;  /* White text for better contrast */
    font-weight: 500;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    border-left: 4px solid #1B5E20;  /* Darker green accent */
}
.translation-container {
    background-color: #1565C0;  /* Blue background */
    border-radius: 10px;
    padding: 20px;
    margin-top: 20px;
    color: #FFFFFF;  /* White text for better contrast */
    font-weight: 500;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    border-left: 4px solid #0D47A1;  /* Darker blue accent */
}
.real-time-transcript {
    background-color: #1565C0;  /* Blue background */
    border-radius: 10px;
    padding: 15px;
    margin-top: 10px;
    min-height: 100px;
    color: #FFFFFF;  /* White text for better contrast */
    font-weight: 500;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    border-left: 4px solid #0D47A1;  /* Darker blue accent */
}
/* Meeting evaluation styles */
.evaluation-container {
    background-color: #5E35B1;  /* Purple background */
    border-radius: 10px;
    padding: 20px;
    margin-top: 20px;
    color: #FFFFFF;  /* White text for better contrast */
    font-weight: 500;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    border-left: 4px solid #4527A0;  /* Darker purple accent */
}
.evaluation-list {
    margin-top: 10px;
    margin-bottom: 15px;
    padding-left: 25px;
}
.evaluation-list li {
    margin-bottom: 12px;
    line-height: 1.6;
    font-size: 1.02em;
}
.evaluation-rating {
    font-size: 1.3em;
    font-weight: bold;
    color: #FFC107;  /* Amber color for rating */
    margin-top: 25px;
    margin-bottom: 15px;
    padding: 8px 15px;
    background-color: rgba(0, 0, 0, 0.2);
    border-radius: 5px;
    display: inline-block;
}
.evaluation-justification {
    font-style: italic;
    margin-bottom: 15px;
    line-height: 1.6;
    padding: 10px;
    background-color: rgba(0, 0, 0, 0.1);
    border-radius: 5px;
    font-size: 1.05em;
}
/* For dark mode support */
@media (prefers-color-scheme: dark) {
    .transcription-container {
        background-color: #2E7D32;  /* Keep same green in dark mode */
        color: #FFFFFF;
        border-left: 4px solid #1B5E20;
    }
    .translation-container {
        background-color: #1565C0;  /* Keep same blue in dark mode */
        color: #FFFFFF;
        border-left: 4px solid #0D47A1;
    }
    .real-time-transcript {
        background-color: #1565C0;  /* Keep same blue in dark mode */
        color: #FFFFFF;
        border-left: 4px solid #0D47A1;
    }
    .evaluation-container {
        background-color: #5E35B1;  /* Keep same purple in dark mode */
        color: #FFFFFF;
        border-left: 4px solid #4527A0;
    }
}

/* Category heading styles */
.category-heading {
    margin-top: 20px;
    margin-bottom: 10px;
}

/* Category styling */
.category-overall {
    background-color: #FBC02D;  /* Amber/Yellow */
    border-radius: 5px;
    padding: 2px 8px;
    margin-right: 10px;
    font-weight: bold;
    color: #333333;  /* Dark text for better contrast on yellow */
}

.category-completed {
    background-color: #2E7D32;  /* Green */
    border-radius: 5px;
    padding: 2px 8px;
    margin-right: 10px;
    font-weight: bold;
}
.category-ongoing {
    background-color: #1565C0;  /* Blue */
    border-radius: 5px;
    padding: 2px 8px;
    margin-right: 10px;
    font-weight: bold;
}
.category-blockers {
    background-color: #C62828;  /* Red */
    border-radius: 5px;
    padding: 2px 8px;
    margin-right: 10px;
    font-weight: bold;
}
.category-ideas {
    background-color: #6A1B9A;  /* Purple */
    border-radius: 5px;
    padding: 2px 8px;
    margin-right: 10px;
    font-weight: bold;
}
.category-todo {
    background-color: #F9A825;  /* Amber */
    border-radius: 5px;
    padding: 2px 8px;
    margin-right: 10px;
    font-weight: bold;
    color: #333333;
}
.category-action {
    background-color: #D84315;  /* Deep Orange */
    border-radius: 5px;
    padding: 2px 8px;
    margin-right: 10px;
    font-weight: bold;
}

/* List formatting for better readability */
.summary-container ul {
    margin-top: 10px;
    margin-bottom: 15px;
    padding-left: 25px;
}

.summary-container li {
    margin-bottom: 5px;
    line-height: 1.5;
}

/* Main summary container styling */
.summary-container {
    background-color: #1A237E;  /* Deep indigo background */
    border-radius: 10px;
    padding: 20px;
    margin-top: 20px;
    color: #FFFFFF;  /* White text for better contrast */
    font-weight: 500;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    border-left: 4px solid #0D1A5E;  /* Darker accent */
}

/* For dark mode support */
@media (prefers-color-scheme: dark) {
    .category-todo, .category-overall {
        color: #333333;  /* Keep dark text on light background */
    }
}

/* Category heading styles */
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='header'>Audio Transcription App</h1>", unsafe_allow_html=True)
st.markdown("Convert speech to text using OpenAI's advanced speech recognition technology")

# API Key input

# Play audio from Sidebar audio folder
audio_path = "Sidebar audio/My name is jeff.mp3"
if os.path.exists(audio_path):
    # Approach 1: Use native Streamlit audio component
    with open(audio_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
        # Add a small, visible audio player in the sidebar
        st.sidebar.audio(audio_bytes, format="audio/mp3", start_time=0)
    
    # Approach 2: Also try HTML with different autoplay techniques
    st.sidebar.markdown(f"""
    <script>
    // Try to play audio after page loads
    document.addEventListener('DOMContentLoaded', (event) => {{
        const audio = document.getElementById('jeffAudio');
        if (audio) {{
            // Add user interaction handler to play audio
            document.body.addEventListener('click', function() {{
                audio.play();
            }}, {{ once: true }});
            
            // Also try to play automatically
            const playPromise = audio.play();
            if (playPromise !== undefined) {{
                playPromise.catch(error => {{
                    console.log("Autoplay prevented by browser");
                }});
            }}
        }}
    }});
    </script>
    <audio id="jeffAudio" autoplay="true" muted="false" style="width:0px; height:0px;">
        <source src="{audio_path}" type="audio/mp3">
    </audio>
    """, unsafe_allow_html=True)
else:
    # Look for any MP3 in the Sidebar audio folder as fallback
    sidebar_dir = "Sidebar audio"
    if os.path.exists(sidebar_dir):
        mp3_files = [f for f in os.listdir(sidebar_dir) if f.endswith('.mp3')]
        if mp3_files:
            fallback_audio = os.path.join(sidebar_dir, mp3_files[0])
            # Use Streamlit's native audio component as fallback
            with open(fallback_audio, "rb") as audio_file:
                audio_bytes = audio_file.read()
                st.sidebar.audio(audio_bytes, format="audio/mp3", start_time=0)

api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
if api_key:
    # Store in session state rather than directly in environment
    if 'openai_api_key' not in st.session_state or st.session_state.openai_api_key != api_key:
        st.session_state.openai_api_key = api_key
        # Also set in environment for backward compatibility
        os.environ["OPENAI_API_KEY"] = api_key

# Add Claude API key input
claude_api_key = st.sidebar.text_input("Enter your Claude API Key (for Claude summaries)", type="password")
if claude_api_key:
    # Store in session state rather than directly in environment
    if 'claude_api_key' not in st.session_state or st.session_state.claude_api_key != claude_api_key:
        st.session_state.claude_api_key = claude_api_key
        # Also set in environment for backward compatibility
        os.environ["CLAUDE_API_KEY"] = claude_api_key

# Add a separator
st.sidebar.markdown("---")

# Navigation section in sidebar
st.sidebar.markdown("## Navigation")

# Initialize the page in session state if it doesn't exist
if 'current_page' not in st.session_state:
    st.session_state.current_page = "main"

# Navigation buttons
if st.sidebar.button("Main App", key="nav_main"):
    st.session_state.current_page = "main"
    
if st.sidebar.button("Things to Add/Do", key="nav_todo"):
    st.session_state.current_page = "todo"

# Check for API key
if not get_api_key("OPENAI_API_KEY") and not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to use this app")
    st.stop()

# Display content based on selected page
if st.session_state.current_page == "todo":
    st.markdown("# Things to Add/Do")
    st.markdown("Use this page to track your tasks and ideas for the app.")
    
    # Initialize todo items in session state if they don't exist
    if 'todo_checkboxes' not in st.session_state:
        st.session_state.todo_checkboxes = {
            "evaluator_mode": False,
            "improve_prompt": False,
            "analyze_compression": False,
            "analyze_formats": False,
            "unified_save": False,
            "notify_ip": False,
            "check_supabase": False,
            "basic_login_then_auth0": False,
            "post_to_slack": False,
            "catchy_headline": False
        }
    
    # Display todo items with checkboxes
    st.markdown("### Your Items")
    
    # Add evaluator mode
    evaluator_checked = st.checkbox(
        "Add evaluator mode",
        value=st.session_state.todo_checkboxes["evaluator_mode"],
        key="checkbox_evaluator"
    )
    st.session_state.todo_checkboxes["evaluator_mode"] = evaluator_checked
    
    # Improve OPENAI prompt
    prompt_checked = st.checkbox(
        "Improve OPENAI prompt",
        value=st.session_state.todo_checkboxes["improve_prompt"],
        key="checkbox_prompt"
    )
    st.session_state.todo_checkboxes["improve_prompt"] = prompt_checked
    
    # Analyze compression vs splitting
    compression_checked = st.checkbox(
        "Analyze compression vs splitting",
        value=st.session_state.todo_checkboxes["analyze_compression"],
        key="checkbox_compression"
    )
    st.session_state.todo_checkboxes["analyze_compression"] = compression_checked
    
    # Analyze conversion of formats
    formats_checked = st.checkbox(
        "Analyze conversion of formats",
        value=st.session_state.todo_checkboxes["analyze_formats"],
        key="checkbox_formats"
    )
    st.session_state.todo_checkboxes["analyze_formats"] = formats_checked
    
    # Create a unified save button
    save_checked = st.checkbox(
        "Create a unified save button",
        value=st.session_state.todo_checkboxes["unified_save"],
        key="checkbox_save"
    )
    st.session_state.todo_checkboxes["unified_save"] = save_checked
    
    # Notify waqas if its office ip
    ip_checked = st.checkbox(
        "Notify waqas if its office ip then it will be restricted and no open will be able to use. Also the file upload limit is 1 GB, not suitable for video recordings ",
        value=st.session_state.todo_checkboxes["notify_ip"],
        key="checkbox_ip"
    )
    st.session_state.todo_checkboxes["notify_ip"] = ip_checked
    
    # Check Supabase for login and memory management
    supabase_checked = st.checkbox(
        "Check if Supabase is easier for login and memory management",
        value=st.session_state.todo_checkboxes["check_supabase"],
        key="checkbox_supabase"
    )
    st.session_state.todo_checkboxes["check_supabase"] = supabase_checked
    
    # Basic login before Auth0
    login_checked = st.checkbox(
        "Check if Supabase, normal if we can start with basic login and then move on to Auth0 and others, this becomes more tricky and annoying",
        value=st.session_state.todo_checkboxes["basic_login_then_auth0"],
        key="checkbox_login"
    )
    st.session_state.todo_checkboxes["basic_login_then_auth0"] = login_checked
    
    # Add Post to Slack button
    slack_checked = st.checkbox(
        "Add a button called Post to Slack, the idea is to integrate it with slack by posting the evaluation and everything to slack",
        value=st.session_state.todo_checkboxes["post_to_slack"],
        key="checkbox_slack"
    )
    st.session_state.todo_checkboxes["post_to_slack"] = slack_checked
    
    # Catchy headline for Waqas
    headline_checked = st.checkbox(
        "Modify the prompt to create a catchy headliner to catch Waqas' attention",
        value=st.session_state.todo_checkboxes["catchy_headline"],
        key="checkbox_headline"
    )
    st.session_state.todo_checkboxes["catchy_headline"] = headline_checked
    
    # Add some helpful info
    st.markdown("---")
    st.info("Check items as you complete them. Your progress will be saved between sessions.")

else:  # Main app content
    # Initialize session state for summaries and transcriptions
    if 'summaries' not in st.session_state:
        st.session_state.summaries = {}
    if 'transcription' not in st.session_state:
        st.session_state.transcription = None
    if 'rt_transcription' not in st.session_state:
        st.session_state.rt_transcription = None
    if 'file_transcription' not in st.session_state:
        st.session_state.file_transcription = None
    # Add a new key in session state for tracking evaluation button clicks
    if 'evaluation_clicked' not in st.session_state:
        st.session_state.evaluation_clicked = False
    if 'meeting_summary_clicked' not in st.session_state:
        st.session_state.meeting_summary_clicked = False
    if 'general_summary_clicked' not in st.session_state:
        st.session_state.general_summary_clicked = False
    
    # Available speech models
    SPEECH_MODELS = {
        "Whisper-1": "whisper-1",
        "GPT-4o Transcribe": "gpt-4o-transcribe",
        "GPT-4o Transcribe Mini": "gpt-4o-mini-transcribe"
    }
    
    # Conditionally create tabs based on environment
    if is_cloud_env or not AUDIO_RECORDING_AVAILABLE:
        # In cloud environment, only show file upload tab
        st.warning("🔴 **Microphone recording and real-time transcription** are disabled in cloud environments. Only file upload is available.")
        # Create a single tab for file upload
        tabs = st.tabs(["File Upload"])
        file_tab = tabs[0]
    else:
        # In local environment, show all tabs
        tabs = st.tabs(["Microphone Recording", "Real-time Transcription", "File Upload"])
        mic_tab, realtime_tab, file_tab = tabs
    
    # Only display microphone tab if we're in a local environment
    if not is_cloud_env and AUDIO_RECORDING_AVAILABLE:
        with mic_tab:
            st.header("Record Audio")
            st.markdown("Use your microphone to record audio for transcription")
            
            col1, col2 = st.columns(2)
            
            with col1:
                duration = st.slider("Recording Duration (seconds)", min_value=5, max_value=60, value=10, step=5)
            
            with col2:
                model_key = st.selectbox(
                    "Select Model",
                    list(SPEECH_MODELS.keys()),
                    index=0
                )
                model = SPEECH_MODELS[model_key]
            
            # Add debug checkbox for testing translation
            show_debug = st.checkbox("Debug mode (force translation option)", value=False, help="Enable this to always show the translation button for testing")
            
            # Record button
            if st.button("Start Recording", type="primary", key="record_btn"):
                with st.spinner("Recording in progress..."):
                    audio_file_path = record_audio(duration)
                    
                    if audio_file_path:
                        st.success("Recording completed!")
                        
                        # Add playback of the recorded audio
                        with open(audio_file_path, "rb") as f:
                            audio_bytes = f.read()
                        
                        st.audio(audio_bytes, format="audio/wav")
                        
                        # Transcribe the audio
                        with st.spinner("Transcribing..."):
                            transcription = transcribe_audio(audio_file_path, model)
                        
                        if transcription:
                            # Store transcription in session state
                            st.session_state.transcription = transcription
                            
                            st.markdown("### Transcription")
                            st.markdown(f"<div class='transcription-container'>{transcription}</div>", unsafe_allow_html=True)
                            
                            # Add download button for transcription
                            get_text_download_link(
                                transcription, 
                                "transcription.txt", 
                                "📥 Download Transcription"
                            )
                            
                            # Check if the transcription is in English
                            # Initialize session state for translation
                            if 'translation_clicked' not in st.session_state:
                                st.session_state.translation_clicked = False
                            if 'translation_result' not in st.session_state:
                                st.session_state.translation_result = None
                            if 'detected_language' not in st.session_state:
                                st.session_state.detected_language = None
                            
                            # Create function to update state without page rerun
                            def translation_callback():
                                st.session_state.translation_clicked = True
                            
                            # Check language and offer translation if needed
                            with st.spinner("Detecting language..."):
                                is_english, _, detected_language = detect_and_translate(transcription)
                                st.session_state.detected_language = detected_language
                            
                            # Always show detected language
                            st.info(f"Detected language: {detected_language}" + 
                                    (" (no translation needed)" if is_english and not show_debug else ""))
                            
                            # Show translate button if not English or if debug mode is enabled
                            if not is_english or show_debug:
                                st.button("Translate to English", on_click=translation_callback, type="primary", key="translate_btn")
                            
                            # Perform translation if button was clicked
                            if st.session_state.translation_clicked:
                                # Handle both cases - when language is detected as non-English,
                                # and when debug mode forces the translation option
                                with st.spinner(f"Translating from {detected_language} to English..."):
                                    # Only translate if we haven't done so already
                                    if not st.session_state.translation_result:
                                        # If it's actually English but we're in debug mode, just use the original text
                                        if is_english and show_debug:
                                            st.session_state.translation_result = "(Debug mode - original text shown as translation)\n\n" + transcription
                                        else:
                                            _, translated_text, _ = detect_and_translate(transcription)
                                            st.session_state.translation_result = translated_text
                            
                                # Display the translation
                                if st.session_state.translation_result:
                                    st.markdown("### English Translation")
                                    st.markdown(f"<div class='translation-container'>{st.session_state.translation_result}</div>", unsafe_allow_html=True)
                                    
                                    # Add download button for translation
                                    get_text_download_link(
                                        st.session_state.translation_result,
                                        "translation.txt",
                                        "📥 Download Translation"
                                    )
                                    
                                    # Reset the translation clicked state for next time
                                    st.session_state.translation_clicked = False
                            
                            # Add summary options
                            st.markdown("### Generate Summary")
                            summary_col1, summary_col2 = st.columns(2)
                            
                            # Create functions to update state without page rerun
                            def meeting_summary_callback():
                                st.session_state.meeting_summary_clicked = True
                            
                            def general_summary_callback():
                                st.session_state.general_summary_clicked = True
                            
                            with summary_col1:
                                st.button("Meeting Conversation", key="meeting_summary_btn", on_click=meeting_summary_callback)
                            
                            with summary_col2:
                                st.button("General Summary", key="general_summary_btn", on_click=general_summary_callback)
                            
                            # Generate Claude Meeting Evaluation automatically if Claude API key is available
                            if get_api_key("CLAUDE_API_KEY"):
                                if 'meeting_evaluation' not in st.session_state.summaries:
                                    with st.spinner("Generating meeting evaluation with Claude 3.7..."):
                                        evaluation = generate_claude_meeting_evaluation(transcription)
                                        if evaluation:
                                            st.session_state.summaries['meeting_evaluation'] = evaluation
                            
                            # Generate and display summaries without page rerun
                            summary_container = st.empty()
                            
                            if st.session_state.meeting_summary_clicked:
                                with st.spinner("Generating meeting summary..."):
                                    # Only generate if it doesn't exist
                                    if 'meeting_summary' not in st.session_state.summaries:
                                        meeting_summary = generate_summary(transcription, "meeting")
                                        if meeting_summary:
                                            st.session_state.summaries['meeting_summary'] = meeting_summary
                                    st.session_state.meeting_summary_clicked = False  # Reset for next time
                                    
                                    # Display the formatted summary
                                    if 'meeting_summary' in st.session_state.summaries:
                                        summary_container.markdown("### Meeting Summary")
                                        # Apply formatting to the meeting summary
                                        formatted_summary = format_meeting_summary(st.session_state.summaries['meeting_summary'])
                                        summary_container.markdown(
                                            f"<div class='summary-container'>{formatted_summary}</div>", 
                                            unsafe_allow_html=True
                                        )
                                        
                                        # Add download button for meeting summary
                                        get_text_download_link(
                                            st.session_state.summaries['meeting_summary'],
                                            "meeting_summary.txt",
                                            "📥 Download Meeting Summary"
                                        )
                                        
                                        # Display the meeting evaluation if available
                                        if 'meeting_evaluation' in st.session_state.summaries:
                                            st.markdown("### Meeting Evaluation (Claude 3.7)")
                                            formatted_evaluation = format_meeting_evaluation(st.session_state.summaries['meeting_evaluation'])
                                            st.markdown(
                                                f"<div class='evaluation-container'>{formatted_evaluation}</div>", 
                                                unsafe_allow_html=True
                                            )
                                            
                                            # Add download button for evaluation
                                            get_text_download_link(
                                                st.session_state.summaries['meeting_evaluation'],
                                                "meeting_evaluation.txt",
                                                "📥 Download Meeting Evaluation"
                                            )
                            
                            if st.session_state.general_summary_clicked:
                                with st.spinner("Generating general summary..."):
                                    # Check if summary already exists in session state
                                    if 'general_summary' not in st.session_state.summaries:
                                        general_summary = generate_summary(transcription, "general")
                                        if general_summary:
                                            st.session_state.summaries['general_summary'] = general_summary
                                    st.session_state.general_summary_clicked = False  # Reset for next time
                                    
                                    # Display the summary
                                    if 'general_summary' in st.session_state.summaries:
                                        summary_container.markdown("### General Summary")
                                        summary_container.markdown(
                                            f"<div class='summary-container'>{st.session_state.summaries['general_summary']}</div>", 
                                            unsafe_allow_html=True
                                        )
                                        
                                        # Add download button for general summary
                                        get_text_download_link(
                                            st.session_state.summaries['general_summary'],
                                            "general_summary.txt",
                                            "📥 Download General Summary"
                                        )
                                        
                                        # Display the meeting evaluation if available
                                        if 'meeting_evaluation' in st.session_state.summaries:
                                            st.markdown("### Meeting Evaluation (Claude 3.7)")
                                            formatted_evaluation = format_meeting_evaluation(st.session_state.summaries['meeting_evaluation'])
                                            st.markdown(
                                                f"<div class='evaluation-container'>{formatted_evaluation}</div>", 
                                                unsafe_allow_html=True
                                            )
                                            
                                            # Add download button for evaluation
                                            get_text_download_link(
                                                st.session_state.summaries['meeting_evaluation'],
                                                "meeting_evaluation.txt",
                                                "📥 Download Meeting Evaluation"
                                            )
                            
                            # Check if we have any summaries from previous runs to display
                            if not st.session_state.meeting_summary_clicked and not st.session_state.general_summary_clicked:
                                if 'meeting_summary' in st.session_state.summaries:
                                    summary_container.markdown("### Meeting Summary")
                                    # Apply formatting to the meeting summary
                                    formatted_summary = format_meeting_summary(st.session_state.summaries['meeting_summary'])
                                    summary_container.markdown(
                                        f"<div class='summary-container'>{formatted_summary}</div>", 
                                        unsafe_allow_html=True
                                    )
                                    
                                    # Add download button for meeting summary from previous run
                                    get_text_download_link(
                                        st.session_state.summaries['meeting_summary'],
                                        "meeting_summary.txt",
                                        "📥 Download Meeting Summary"
                                    )
                                    
                                    # Display the meeting evaluation if available
                                    if 'meeting_evaluation' in st.session_state.summaries:
                                        st.markdown("### Meeting Evaluation (Claude 3.7)")
                                        formatted_evaluation = format_meeting_evaluation(st.session_state.summaries['meeting_evaluation'])
                                        st.markdown(
                                            f"<div class='evaluation-container'>{formatted_evaluation}</div>", 
                                            unsafe_allow_html=True
                                        )
                                        
                                        # Add download button for evaluation
                                        get_text_download_link(
                                            st.session_state.summaries['meeting_evaluation'],
                                            "meeting_evaluation.txt",
                                            "📥 Download Meeting Evaluation"
                                        )
                                elif 'general_summary' in st.session_state.summaries:
                                    summary_container.markdown("### General Summary")
                                    summary_container.markdown(
                                        f"<div class='summary-container'>{st.session_state.summaries['general_summary']}</div>", 
                                        unsafe_allow_html=True
                                    )
                                    
                                    # Add download button for general summary from previous run
                                    get_text_download_link(
                                        st.session_state.summaries['general_summary'],
                                        "general_summary.txt",
                                        "📥 Download General Summary"
                                    )
                                    
                                    # Display the meeting evaluation if available
                                    if 'meeting_evaluation' in st.session_state.summaries:
                                        st.markdown("### Meeting Evaluation (Claude 3.7)")
                                        formatted_evaluation = format_meeting_evaluation(st.session_state.summaries['meeting_evaluation'])
                                        st.markdown(
                                            f"<div class='evaluation-container'>{formatted_evaluation}</div>", 
                                            unsafe_allow_html=True
                                        )
                                        
                                        # Add download button for evaluation
                                        get_text_download_link(
                                            st.session_state.summaries['meeting_evaluation'],
                                            "meeting_evaluation.txt",
                                            "📥 Download Meeting Evaluation"
                                        )
                            
                            # Clean up temporary file
                            os.unlink(audio_file_path)
                        else:
                            st.error("Recording failed. Please check your microphone permissions.")

    # Only display real-time tab if we're in a local environment
    if not is_cloud_env and AUDIO_RECORDING_AVAILABLE:
        with realtime_tab:
            st.header("Enhanced Real-time Transcription")
            st.markdown("Transcribe and translate audio in real-time with chunked processing")
            
            # Initialize session state for chunked streaming
            if 'chunked_streaming' not in st.session_state:
                st.session_state.chunked_streaming = False
                st.session_state.audio_processor = None
            
            # Display any connection errors
            if 'realtime_error' in st.session_state and st.session_state.realtime_error:
                st.error(f"Error: {st.session_state.realtime_error}")
                st.session_state.realtime_error = None
            
            # Add debug checkbox for testing
            show_debug_realtime = st.checkbox("Debug mode (shows more info)", value=False, 
                                   help="Enable this for troubleshooting connection issues", key="debug_new_realtime")
            
            # Add translation toggle
            show_translation = st.checkbox("Enable automatic translation", value=True,
                                help="Automatically translate non-English speech to English")
            
            # Start/Stop streaming buttons
            col1, col2 = st.columns(2)
            with col1:
                start_button = st.button("Start Streaming", type="primary", 
                                        disabled=st.session_state.chunked_streaming, 
                                        key="start_chunked_realtime")
            with col2:
                stop_button = st.button("Stop Streaming", type="secondary", 
                                       disabled=not st.session_state.chunked_streaming, 
                                       key="stop_chunked_realtime")
            
            # Create placeholders for transcript and translation
            realtime_transcript_container = st.empty()
            realtime_translation_container = st.empty()
            
            if start_button:
                # Get API key from secrets, environment, or user input
                api_key = get_api_key("OPENAI_API_KEY") or st.session_state.get('openai_api_key')
                if not api_key:
                    st.error("API key is required for real-time transcription")
                else:
                    try:
                        # Initialize audio processor
                        st.session_state.audio_processor = ChunkedAudioProcessor(api_key)
                        st.session_state.chunked_streaming = True
                        
                        # Set up audio stream with sounddevice
                        if not AUDIO_RECORDING_AVAILABLE or sd is None:
                            st.error("Microphone recording is not available in this environment")
                        else:
                            # Audio callback function for streaming
                            def audio_callback(indata, frames, time_info, status):
                                """Callback to capture audio data from microphone"""
                                if status:
                                    print(f"Audio status: {status}")
                                
                                # Convert float32 audio data to int16
                                audio_data = (indata * 32767).astype(np.int16).tobytes()
                                
                                # Add to processor
                                if st.session_state.audio_processor:
                                    st.session_state.audio_processor.add_audio_chunk(audio_data)
                            
                            # Start stream
                            stream = sd.InputStream(
                                samplerate=16000,
                                channels=1,
                                callback=audio_callback
                            )
                            stream.start()
                            
                            # Store stream in session state
                            st.session_state.audio_stream = stream
                            
                            realtime_transcript_container.markdown(
                                "<div class='real-time-transcript'>Listening... (Chunked processing active)</div>", 
                                unsafe_allow_html=True
                            )
                            
                            if show_debug_realtime:
                                st.success("Audio streaming started successfully!")
                
                    except Exception as e:
                        st.session_state.realtime_error = str(e)
                        st.session_state.chunked_streaming = False
                        st.session_state.audio_processor = None
                        if show_debug_realtime:
                            st.error(f"Exception details: {e}")
                        st.rerun()
                    
                    # Rerun to update UI state
                    st.rerun()
            
            if stop_button and st.session_state.chunked_streaming:
                if 'audio_stream' in st.session_state and st.session_state.audio_stream:
                    st.session_state.audio_stream.stop()
                    st.session_state.audio_stream.close()
                    
                # Get final transcript and translation
                final_transcript = ""
                final_translation = ""
                detected_language = "Unknown"
                
                if st.session_state.audio_processor:
                    final_transcript = st.session_state.audio_processor.get_transcript()
                    final_translation = st.session_state.audio_processor.get_translation()
                    detected_language = st.session_state.audio_processor.get_language()
                
                # Display final transcript
                if final_transcript:
                    realtime_transcript_container.markdown(
                        f"<div class='real-time-transcript'>{final_transcript}</div>", 
                        unsafe_allow_html=True
                    )
                    
                    # Add download button for real-time transcription
                    get_text_download_link(
                        final_transcript,
                        "realtime_transcription.txt",
                        "📥 Download Transcription"
                    )
                    
                    # Display language detection and translation if available
                    if detected_language != "Unknown":
                        st.info(f"Detected language: {detected_language}")
                        
                        if detected_language != "English" and final_translation:
                            realtime_translation_container.markdown(
                                f"<div class='translation-container'>{final_translation}</div>", 
                                unsafe_allow_html=True
                            )
                            
                            # Add download button for translation
                            get_text_download_link(
                                final_translation,
                                "realtime_translation.txt",
                                "📥 Download Translation"
                            )
                    
                    # Store in session state for summary generation
                    st.session_state.rt_transcription = final_transcript
                
                # Reset session state
                st.session_state.chunked_streaming = False
                st.session_state.audio_processor = None
                
                # Success message
                st.success("Transcription completed!")
                
                # Show summary options after stopping
                if final_transcript:
                    # Add summary options
                    st.markdown("### Generate Summary")
                    # [Summary options code as before]
                
                # Rerun to update UI state
                st.rerun()
            
            # Update the transcript in real-time if streaming
            if st.session_state.chunked_streaming and st.session_state.audio_processor:
                # Get current transcript and translation
                current_transcript = st.session_state.audio_processor.get_transcript()
                current_translation = st.session_state.audio_processor.get_translation()
                current_language = st.session_state.audio_processor.get_language()
                
                # Update transcript display
                if current_transcript:
                    realtime_transcript_container.markdown(
                        f"<div class='real-time-transcript'>{current_transcript}</div>", 
                        unsafe_allow_html=True
                    )
                    
                    # Show translation if available and enabled
                    if show_translation and current_language not in ["English", "Unknown"] and current_translation:
                        realtime_translation_container.markdown(
                            f"<div class='translation-container'>{current_translation}</div>", 
                            unsafe_allow_html=True
                        )
                        
                        # Show detected language in debug mode
                        if show_debug_realtime:
                            st.info(f"Detected language: {current_language}")
                
                # Show debug info if enabled
                if show_debug_realtime:
                    st.info(f"Audio chunk size: {st.session_state.audio_processor.chunk_duration_ms}ms | Processing active: {st.session_state.audio_processor.is_processing}")
                
                # Add automatic rerun for real-time updates
                time.sleep(0.3)  # Brief pause (shorter than chunk size)
                st.rerun()

    # File upload tab (available in all environments)
    with file_tab:
        st.header("Upload Audio File")
        st.markdown("Upload an MP3 or MP4 file for transcription or translation")
        
        # Model selection
        file_model_key = st.selectbox(
            "Select Model",
            list(SPEECH_MODELS.keys()),
            index=0,
            key="file_model"
        )
        file_model = SPEECH_MODELS[file_model_key]
        
        # Show translation option only when Whisper model is selected
        translate = False
        if file_model_key == "Whisper-1":
            translate = st.checkbox(
                "Translate to English instead of transcribing in original language", 
                help="Uses OpenAI's translations API to translate audio in any language to English text"
            )
        
        # Add option for Claude summary
        use_claude = st.checkbox(
            "Also generate summary using Claude 3.7", 
            help="In addition to the standard GPT summary, also generate a summary using Claude 3.7 with more detailed bullet points and enhanced formatting"
        )
        
        # File uploader with note about large file support
        uploaded_file = st.file_uploader(
            "Choose an audio or video file", 
            type=["mp3", "mp4", "wav", "m4a"],
            help="Supports files of any size with automatic MP4 to MP3 conversion"
        )
        
        if uploaded_file is not None:
            # Create temporary file
            file_extension = uploaded_file.name.split(".")[-1].lower()
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}')
            temp_file_path = temp_file.name
            temp_file.write(uploaded_file.getvalue())
            temp_file.close()
            
            # Display file
            if file_extension == "mp4":
                # For MP4 videos, display the video player
                st.video(uploaded_file)
                st.info("📽️ Video file detected: Audio will be automatically extracted for transcription.")
            else:
                # For audio files, display the audio player
                st.audio(uploaded_file, format=f"audio/{file_extension}")
            
            # Transcribe/Translate and Summarize button
            button_text = "Translate and Summarize" if translate else "Transcribe and Summarize"
            if st.button(button_text, type="primary", key="transcribe_file"):
                # For videos and m4a files, first extract the audio or convert to mp3
                audio_file_for_transcription = temp_file_path
                temp_files_to_clean = [temp_file_path]  # Track files to clean up
                
                if file_extension == "mp4":
                    with st.spinner("Extracting audio from video..."):
                        audio_file_for_transcription, file_size_mb, duration_formatted = extract_audio_from_video(temp_file_path)
                        if audio_file_for_transcription:
                            temp_files_to_clean.append(audio_file_for_transcription)
                            
                            # Display success message with file details
                            st.success(f"✅ Audio extracted successfully! MP3 file size: {file_size_mb:.2f} MB, Duration: {duration_formatted}")
                            
                            # Offer download option for the extracted audio file
                            with open(audio_file_for_transcription, "rb") as audio_file:
                                mp3_bytes = audio_file.read()
                                original_filename = uploaded_file.name.rsplit('.', 1)[0]
                                st.download_button(
                                    label="Download Extracted MP3",
                                    data=mp3_bytes,
                                    file_name=f"{original_filename}.mp3",
                                    mime="audio/mp3",
                                    help="Download the extracted audio file for future use"
                                )
                        else:
                            st.error("Failed to extract audio from video. Please try uploading an audio file directly.")
                            # Clean up the original temp file
                            os.unlink(temp_file_path)
                            st.stop()
                # Handle m4a files by converting to mp3 first to avoid format compatibility issues
                elif file_extension == "m4a":
                    with st.spinner("Converting audio format..."):
                        try:
                            # Load the audio file
                            audio = AudioSegment.from_file(temp_file_path, format="m4a")
                            
                            # Create a temporary MP3 file
                            temp_mp3_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                            temp_mp3_path = temp_mp3_file.name
                            temp_mp3_file.close()
                            
                            # Export to MP3
                            audio.export(temp_mp3_path, format="mp3", bitrate="192k")
                            
                            # Update the audio file path for transcription
                            audio_file_for_transcription = temp_mp3_path
                            temp_files_to_clean.append(temp_mp3_path)
                            
                            # Get file size and duration
                            file_size_mb = os.path.getsize(temp_mp3_path) / (1024 * 1024)
                            duration_seconds = len(audio) / 1000
                            hours = int(duration_seconds // 3600)
                            minutes = int((duration_seconds % 3600) // 60)
                            seconds = int(duration_seconds % 60)
                            duration_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                            
                            st.success(f"✅ Audio converted successfully! MP3 file size: {file_size_mb:.2f} MB, Duration: {duration_formatted}")
                        except Exception as e:
                            st.error(f"Failed to convert M4A to MP3: {str(e)}")
                            # Clean up the original temp file
                            os.unlink(temp_file_path)
                            st.stop()
                
                # Proceed with transcription using the audio file
                with st.spinner(f"{'Translating' if translate else 'Transcribing'} audio..."):
                    transcription = transcribe_audio(audio_file_for_transcription, file_model, translate)
                
                if transcription:
                    # Store transcription in session state
                    st.session_state.file_transcription = transcription
                    
                    st.markdown(f"### {'Translation' if translate else 'Transcription'}")
                    st.markdown(f"<div class='transcription-container'>{transcription}</div>", unsafe_allow_html=True)
                    
                    # Add download button for file transcription/translation
                    download_filename = "translation.txt" if translate else "transcription.txt"
                    get_text_download_link(
                        transcription,
                        download_filename,
                        f"📥 Download {'Translation' if translate else 'Transcription'}"
                    )
                    
                    # Only check for language and offer translation if not already translating
                    if not translate:
                        # Initialize session state for translation in file upload tab
                        if 'file_translation_clicked' not in st.session_state:
                            st.session_state.file_translation_clicked = False
                        if 'file_translation_result' not in st.session_state:
                            st.session_state.file_translation_result = None
                        if 'file_detected_language' not in st.session_state:
                            st.session_state.file_detected_language = None
                        
                        # Create function to update state without page rerun
                        def file_translation_callback():
                            st.session_state.file_translation_clicked = True
                        
                        # Check language and offer translation if needed
                        with st.spinner("Detecting language..."):
                            is_english, _, detected_language = detect_and_translate(transcription)
                            st.session_state.file_detected_language = detected_language
                        
                        if not is_english:
                            st.info(f"Detected language: {detected_language}")
                            # Show a translate button if not English
                            st.button("Translate to English", on_click=file_translation_callback, type="primary", key="file_translate_btn")
                        else:
                            st.info(f"Detected language: {detected_language} (no translation needed)")
                        
                        # Perform translation if button was clicked
                        if st.session_state.file_translation_clicked and not is_english:
                            with st.spinner(f"Translating from {detected_language} to English..."):
                                # Only translate if we haven't done so already
                                if not st.session_state.file_translation_result:
                                    _, translated_text, _ = detect_and_translate(transcription)
                                    st.session_state.file_translation_result = translated_text
                            
                            # Display the translation
                            if st.session_state.file_translation_result:
                                st.markdown("### English Translation")
                                st.markdown(f"<div class='translation-container'>{st.session_state.file_translation_result}</div>", unsafe_allow_html=True)
                                
                                # Add download button for translation
                                get_text_download_link(
                                    st.session_state.file_translation_result,
                                    "translation.txt",
                                    "📥 Download Translation"
                                )
                                
                                # Reset the translation clicked state for next time
                                st.session_state.file_translation_clicked = False
                    
                    # Create tabs for GPT and Claude summaries
                    if use_claude:
                        summary_tabs = st.tabs(["GPT4.1 Summary", "Claude 3.7 Summary", "Meeting Evaluation"])
                        
                        with summary_tabs[0]:
                            # Automatically generate meeting summary with GPT
                            with st.spinner("Generating meeting summary with GPT-4.1 ..."):
                                meeting_summary = generate_summary(transcription, "meeting")
                                if meeting_summary:
                                    st.markdown("### Meeting Summary (GPT-4o Mini)")
                                    # Apply formatting to the meeting summary
                                    formatted_summary = format_meeting_summary(meeting_summary)
                                    st.markdown(f"<div class='summary-container'>{formatted_summary}</div>", unsafe_allow_html=True)
                        
                                    # Add download button for GPT meeting summary
                                    get_text_download_link(
                                        meeting_summary,
                                        "gpt_meeting_summary.txt",
                                        "📥 Download GPT Meeting Summary"
                                    )
                        
                            # Automatically generate general summary with GPT
                            with st.spinner("Generating general summary with GPT-4o Mini..."):
                                general_summary = generate_summary(transcription, "general")
                                if general_summary:
                                    st.markdown("### General Summary (GPT-4o Mini)")
                                    st.markdown(f"<div class='summary-container'>{general_summary}</div>", unsafe_allow_html=True)
                                    
                                    # Add download button for GPT general summary
                                    get_text_download_link(
                                        general_summary,
                                        "gpt_general_summary.txt",
                                        "📥 Download GPT General Summary"
                                    )
                        
                        with summary_tabs[1]:
                            # Generate meeting summary with Claude
                            with st.spinner("Generating meeting summary with Claude 3.7..."):
                                # Check if Claude API key is available before attempting to generate summary
                                if not get_api_key("CLAUDE_API_KEY"):
                                    st.warning("Claude API key is required. Please enter it in the sidebar.")
                                else:
                                    claude_meeting_summary = generate_claude_summary(transcription, "meeting")
                                    if claude_meeting_summary:
                                        st.markdown("### Meeting Summary (Claude 3.7)")
                                        # Apply formatting to the Claude meeting summary
                                        formatted_claude_summary = format_nyt_summary(claude_meeting_summary)
                                        st.markdown(f"<div class='summary-container'>{formatted_claude_summary}</div>", unsafe_allow_html=True)
                                        
                                        # Add download button for Claude meeting summary
                                        get_text_download_link(
                                            claude_meeting_summary,
                                            "claude_meeting_summary.txt",
                                            "📥 Download Claude Meeting Summary"
                                        )
                                    else:
                                        st.error("Failed to generate Claude meeting summary. Please check the console for errors.")
                            
                            # Generate general summary with Claude
                            with st.spinner("Generating general summary with Claude 3.7..."):
                                # Check if Claude API key is available before attempting to generate summary
                                if not get_api_key("CLAUDE_API_KEY"):
                                    st.warning("Claude API key is required. Please enter it in the sidebar.")
                                else:
                                    claude_general_summary = generate_claude_summary(transcription, "general")
                                    if claude_general_summary:
                                        st.markdown("### General Summary (Claude 3.7)")
                                        st.markdown(f"<div class='summary-container'>{claude_general_summary}</div>", unsafe_allow_html=True)
                                        
                                        # Add download button for Claude general summary
                                        get_text_download_link(
                                            claude_general_summary,
                                            "claude_general_summary.txt",
                                            "📥 Download Claude General Summary"
                                        )
                                    else:
                                        st.error("Failed to generate Claude general summary. Please check the console for errors.")
                        
                        with summary_tabs[2]:
                            # Automatically generate meeting evaluation with Claude
                            with st.spinner("Generating meeting evaluation with Claude 3.7..."):
                                # Check if Claude API key is available
                                if not get_api_key("CLAUDE_API_KEY"):
                                    st.warning("Claude API key is required. Please enter it in the sidebar.")
                                else:
                                    # Generate evaluation
                                    evaluation = generate_claude_meeting_evaluation(transcription)
                                    if evaluation:
                                        st.markdown("### Meeting Evaluation (Claude 3.7)")
                                        # Display the formatted evaluation
                                        formatted_evaluation = format_meeting_evaluation(evaluation)
                                        st.markdown(f"<div class='evaluation-container'>{formatted_evaluation}</div>", unsafe_allow_html=True)
                                        
                                        # Add download button for evaluation
                                        get_text_download_link(
                                            evaluation,
                                            "meeting_evaluation.txt",
                                            "📥 Download Meeting Evaluation"
                                        )
                    else:
                        # Original behavior without Claude option
                        # Automatically generate meeting summary
                        with st.spinner("Generating meeting summary..."):
                            meeting_summary = generate_summary(transcription, "meeting")
                            if meeting_summary:
                                st.markdown("### Meeting Summary")
                                # Apply formatting to the meeting summary
                                formatted_summary = format_meeting_summary(meeting_summary)
                                st.markdown(f"<div class='summary-container'>{formatted_summary}</div>", unsafe_allow_html=True)
                                
                                # Add download button for meeting summary
                                get_text_download_link(
                                    meeting_summary,
                                    "meeting_summary.txt",
                                    "📥 Download Meeting Summary"
                                )
                        
                        # Automatically generate general summary
                        with st.spinner("Generating general summary..."):
                            general_summary = generate_summary(transcription, "general")
                        if general_summary:
                            st.markdown("### General Summary")
                            st.markdown(f"<div class='summary-container'>{general_summary}</div>", unsafe_allow_html=True)
                            
                            # Add download button for general summary
                            get_text_download_link(
                                general_summary,
                                "general_summary.txt",
                                "📥 Download General Summary"
                            )
                    
                    # Remove the separate Meeting Evaluation button section since it's now integrated in the tabs
                    # Clean up all temporary files
                    for file_path in temp_files_to_clean:
                        if os.path.exists(file_path):
                            os.unlink(file_path)
                            print(f"Cleaned up temporary file: {file_path}")

# Sidebar info
with st.sidebar:
    st.markdown("## About")
    
    # Add environment indicator
    if is_cloud_env:
        st.info("📡 **Cloud Environment Detected**\nRunning on Streamlit Cloud - microphone access is disabled.")
    else:
        st.success("🖥️ **Local Environment Detected**\nAll features are available.")
    
    st.markdown("""
    This app uses OpenAI's speech recognition technology to transcribe and translate audio.
    
    ### Available Models:
    - **Whisper-1**: OpenAI's standard speech recognition model
      - Supports both transcription and translation to English
      - Handles large files (>24.5MB) by automatically splitting into smaller chunks
      - Memory-efficient processing for very large audio files
    - **GPT-4o Transcribe**: Latest model with real-time capabilities
    - **GPT-4o Transcribe Mini**: Lighter and faster real-time transcription model
    - **Claude 3 Sonnet**: Used for generating alternative AI summaries when the Claude option is selected
    
    ### Features:
    """)
    
    # Display feature availability based on environment
    if not is_cloud_env and AUDIO_RECORDING_AVAILABLE:
        st.markdown("""
        ✅ Record audio via microphone
        ✅ Real-time streaming transcription with WebSockets
        ✅ Upload audio files for transcription
        ✅ Translate non-English audio to English text
        ✅ Process large audio files by automatically splitting them
        ✅ Generate meeting summaries with multiple AI models
        """)
    else:
        st.markdown("""
        ❌ Record audio via microphone (unavailable in cloud)
        ❌ Real-time streaming transcription (unavailable in cloud)
        ✅ Upload audio files for transcription
        ✅ Translate non-English audio to English text
        ✅ Process large audio files by automatically splitting them
        ✅ Generate meeting summaries with multiple AI models
        """)
    
    st.markdown("""
    ### Technical Details:
    The real-time transcription uses OpenAI's Realtime API with WebSockets to process audio streams as you speak, providing instant transcription using the GPT-4o Transcribe model. This feature requires a local environment with audio hardware.
    
    File upload works in all environments, including Streamlit Cloud.
    """)

# Add additional note about deployment constraints
if is_cloud_env:
    st.markdown("""
    <div style='background-color: #FFF3CD; color: #856404; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 5px solid #FFECB5;'>
        <h3 style='margin-top: 0;'>⚠️ Cloud Deployment Notice</h3>
        <p>This app is running on Streamlit Cloud, which doesn't support audio hardware access.</p>
        <p><strong>Limited functionality:</strong> Microphone recording and real-time transcription are disabled.</p>
        <p><strong>Available features:</strong> File upload, transcription, translation, and summary generation with GPT-4 and Claude 3.7 remain fully functional.</p>
        <p>For full functionality including microphone access, run this app locally.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and OpenAI's speech recognition technology | [Privacy Policy](/privacy)", unsafe_allow_html=True)

