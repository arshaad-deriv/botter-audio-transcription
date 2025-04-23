# Set environment variable to prevent pygame from initializing GUI components
# This fixes the "setting the main menu on a non-main thread" error on macOS
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import streamlit as st
import tempfile
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import openai
import anthropic
from pydub import AudioSegment
from dotenv import load_dotenv
import base64
import queue
import threading
import io
import json
import websocket
import audioop
import struct
from moviepy.editor import VideoFileClip  # For MP4 to MP3 conversion
import math

# Load environment variables
load_dotenv()

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
            
            # Create temporary file for the chunk
            temp_chunk_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
            chunk_path = temp_chunk_file.name
            temp_chunk_file.close()
            
            # Export chunk to file
            chunk.export(chunk_path, format=file_extension.replace('.', ''))
            
            # Add to list of chunk paths
            chunk_paths.append(chunk_path)
        
        return chunk_paths
    
    except Exception as e:
        st.error(f"Error splitting audio file: {str(e)}")
        print(f"Audio splitting error: {str(e)}")
        return [audio_file_path]  # Return original file if splitting fails

# Function to record audio from microphone
def record_audio(duration):
    """Record audio from microphone for specified duration"""
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

# Function to transcribe audio using OpenAI API
def transcribe_audio(audio_file_path, model="whisper-1", translate=False):
    """Transcribe audio file using OpenAI's API"""
    try:
        # Create a client instance
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
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
        
        # Write audio to MP3 file
        audio.write_audiofile(temp_audio_path, codec='mp3', bitrate='192k')
        
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
        if "error" in response:
            self.error_message = response["error"]
            print(f"WebSocket error: {response['error']}")
        
        if "text" in response:
            self.transcript = response["text"]
    
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
                        "audio": audio_base64,
                        "internal": "true"
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
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Initialize WebSocket connection
        self.ws = websocket.WebSocketApp(
            "wss://online-audio.openai.com/v1/transcribe",
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
    
    def stop(self):
        self.stop_event.set()
        if self.ws:
            self.ws.close()
    
    def get_transcript(self):
        return self.transcript

# Function to generate a summary using GPT-4
def generate_summary(text, summary_type):
    """Generate a summary of the transcription using OpenAI's GPT models"""
    try:
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
        
        # Make API call
        response = openai.chat.completions.create(
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
        claude_key = os.environ.get("CLAUDE_API_KEY")
        if not claude_key:
            st.warning("Claude API key is required for summaries. Please enter it in the sidebar.")
            return None
        
        # Initialize client with Claude API key
        claude_client = anthropic.Anthropic(api_key=claude_key)
        
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
        
        # Call Claude API
        response = claude_client.messages.create(
            model="claude-3-7-sonnet-latest",
            max_tokens=4000,
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
        claude_key = os.environ.get("CLAUDE_API_KEY")
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

# Set page configuration
st.set_page_config(
    page_title="Audio Transcription App",
    page_icon="🎙️",
    layout="wide",
)

# Custom CSS
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
    os.environ["OPENAI_API_KEY"] = api_key
    openai.api_key = api_key

# Add Claude API key input
claude_api_key = st.sidebar.text_input("Enter your Claude API Key (for Claude summaries)", type="password")
if claude_api_key:
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
if not os.environ.get("OPENAI_API_KEY") and not api_key:
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
    
    # Tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["Microphone Recording", "Real-time Transcription", "File Upload"])

    with tab1:
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
        
        # Record button
        if st.button("Start Recording", type="primary", key="record_btn"):
            with st.spinner("Recording in progress..."):
                audio_file_path = record_audio(duration)
                
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
                    if os.environ.get("CLAUDE_API_KEY"):
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

    with tab2:
        st.header("Real-time Transcription with GPT-4o")
        st.markdown("Transcribe audio in real-time using OpenAI's Realtime API and gpt-4o-transcribe model")
        
        # Initialize session state for streaming
        if 'realtime_streaming' not in st.session_state:
            st.session_state.realtime_streaming = False
            st.session_state.realtime_transcriber = None
            st.session_state.ws_error = None
        
        # Display any WebSocket connection errors
        if st.session_state.get('ws_error'):
            st.error(f"WebSocket Error: {st.session_state.ws_error}")
            st.session_state.ws_error = None
        
        # Start/Stop streaming buttons
        col1, col2 = st.columns(2)
        with col1:
            start_button = st.button("Start Streaming", type="primary", disabled=st.session_state.realtime_streaming, key="start_realtime")
        with col2:
            stop_button = st.button("Stop Streaming", type="secondary", disabled=not st.session_state.realtime_streaming, key="stop_realtime")
        
        # Transcript placeholder
        realtime_transcript_container = st.empty()
        
        if start_button:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                st.error("API key is required for real-time transcription")
            else:
                st.session_state.realtime_streaming = True
                st.session_state.realtime_transcriber = RealtimeTranscription(api_key)
                
                try:
                    st.session_state.realtime_transcriber.start()
                    realtime_transcript_container.markdown("<div class='real-time-transcript'>Listening... (WebSocket connection with GPT-4o Transcribe)</div>", unsafe_allow_html=True)
                    
                    # Check if connection was successful after a short delay
                    time.sleep(2)
                    if not st.session_state.realtime_transcriber.is_connected:
                        error = st.session_state.realtime_transcriber.get_error() or "Failed to establish WebSocket connection"
                        st.session_state.ws_error = f"{error}. Please check your API key and try again."
                        st.session_state.realtime_streaming = False
                        st.session_state.realtime_transcriber.stop()
                        st.session_state.realtime_transcriber = None
                        st.rerun()
                except Exception as e:
                    st.session_state.ws_error = str(e)
                    st.session_state.realtime_streaming = False
                    st.rerun()
                
                # Rerun to update UI state
                st.rerun()
        
        if stop_button and st.session_state.realtime_streaming:
            if st.session_state.realtime_transcriber:
                st.session_state.realtime_transcriber.stop()
                final_transcript = st.session_state.realtime_transcriber.get_transcript()
                realtime_transcript_container.markdown(f"<div class='real-time-transcript'>{final_transcript}</div>", unsafe_allow_html=True)
                st.session_state.realtime_streaming = False
                
                # Add download button for real-time transcription
                get_text_download_link(
                    final_transcript,
                    "realtime_transcription.txt",
                    "📥 Download Transcription"
                )
                
                # Store transcription in session state
                st.session_state.rt_transcription = final_transcript
                
                # Success message
                st.success("Transcription completed!")
                
                # Add summary options if transcript is available
                if final_transcript:
                    st.markdown("### Generate Summary")
                    rt_summary_col1, rt_summary_col2 = st.columns(2)
                    rt_summary_container = st.empty()
                    
                    # Add keys to session state to track button clicks without page rerun
                    if 'rt_meeting_summary_clicked' not in st.session_state:
                        st.session_state.rt_meeting_summary_clicked = False
                    if 'rt_general_summary_clicked' not in st.session_state:
                        st.session_state.rt_general_summary_clicked = False
                    
                    # Create functions to update state without page rerun
                    def rt_meeting_summary_callback():
                        st.session_state.rt_meeting_summary_clicked = True
                    
                    def rt_general_summary_callback():
                        st.session_state.rt_general_summary_clicked = True
                    
                    with rt_summary_col1:
                        st.button("Meeting Conversation", key="rt_meeting_summary_btn", on_click=rt_meeting_summary_callback)
                    
                    with rt_summary_col2:
                        st.button("General Summary", key="rt_general_summary_btn", on_click=rt_general_summary_callback)
                    
                    # Generate and display summaries without page rerun
                    if st.session_state.rt_meeting_summary_clicked:
                        with st.spinner("Generating meeting summary..."):
                            # Check if summary already exists in session state
                            if 'rt_meeting_summary' not in st.session_state.summaries:
                                meeting_summary = generate_summary(final_transcript, "meeting")
                                if meeting_summary:
                                    st.session_state.summaries['rt_meeting_summary'] = meeting_summary
                            st.session_state.rt_meeting_summary_clicked = False  # Reset for next time
                            
                            # Display the formatted summary
                            if 'rt_meeting_summary' in st.session_state.summaries:
                                rt_summary_container.markdown("### Meeting Summary")
                                # Apply formatting to the meeting summary
                                formatted_summary = format_meeting_summary(st.session_state.summaries['rt_meeting_summary'])
                                rt_summary_container.markdown(
                                    f"<div class='summary-container'>{formatted_summary}</div>", 
                                    unsafe_allow_html=True
                                )
                                
                                # Add download button for meeting summary
                                get_text_download_link(
                                    st.session_state.summaries['rt_meeting_summary'],
                                    "realtime_meeting_summary.txt",
                                    "📥 Download Meeting Summary"
                                )
                    
                    if st.session_state.rt_general_summary_clicked:
                        with st.spinner("Generating general summary..."):
                            # Check if summary already exists in session state
                            if 'rt_general_summary' not in st.session_state.summaries:
                                general_summary = generate_summary(final_transcript, "general")
                                if general_summary:
                                    st.session_state.summaries['rt_general_summary'] = general_summary
                            st.session_state.rt_general_summary_clicked = False  # Reset for next time
                            
                            # Display the summary
                            if 'rt_general_summary' in st.session_state.summaries:
                                rt_summary_container.markdown("### General Summary")
                                rt_summary_container.markdown(
                                    f"<div class='summary-container'>{st.session_state.summaries['rt_general_summary']}</div>", 
                                    unsafe_allow_html=True
                                )
                                
                                # Add download button for general summary from previous run
                                get_text_download_link(
                                    st.session_state.summaries['rt_general_summary'],
                                    "realtime_general_summary.txt",
                                    "📥 Download General Summary"
                                )
                    
                    # Check if we have any summaries from previous runs to display
                    if not st.session_state.rt_meeting_summary_clicked and not st.session_state.rt_general_summary_clicked:
                        if 'rt_meeting_summary' in st.session_state.summaries:
                            rt_summary_container.markdown("### Meeting Summary")
                            # Apply formatting to the meeting summary
                            formatted_summary = format_meeting_summary(st.session_state.summaries['rt_meeting_summary'])
                            rt_summary_container.markdown(
                                f"<div class='summary-container'>{formatted_summary}</div>", 
                                unsafe_allow_html=True
                            )
                            
                            # Add download button for meeting summary from previous run
                            get_text_download_link(
                                st.session_state.summaries['rt_meeting_summary'],
                                "realtime_meeting_summary.txt",
                                "📥 Download Meeting Summary"
                            )
                        elif 'rt_general_summary' in st.session_state.summaries:
                            rt_summary_container.markdown("### General Summary")
                            rt_summary_container.markdown(
                                f"<div class='summary-container'>{st.session_state.summaries['rt_general_summary']}</div>", 
                                unsafe_allow_html=True
                            )
                            
                            # Add download button for general summary from previous run
                            get_text_download_link(
                                st.session_state.summaries['rt_general_summary'],
                                "realtime_general_summary.txt",
                                "📥 Download General Summary"
                            )
                
                # Rerun to update UI state
                st.rerun()
        
        # Update the transcript in real-time if streaming
        if st.session_state.realtime_streaming and st.session_state.realtime_transcriber:
            # Update every second
            current_transcript = st.session_state.realtime_transcriber.get_transcript()
            if current_transcript:
                realtime_transcript_container.markdown(f"<div class='real-time-transcript'>{current_transcript}</div>", unsafe_allow_html=True)
            
            # Add automatic rerun for real-time updates
            time.sleep(0.5)  # Brief pause
            st.rerun()

    with tab3:
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
                # For videos, first extract the audio
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
                    
                    # Create tabs for GPT and Claude summaries
                    if use_claude:
                        summary_tabs = st.tabs(["o3-mini Summary", "Claude 3.7 Summary", "Meeting Evaluation"])
                        
                        with summary_tabs[0]:
                            # Automatically generate meeting summary with GPT
                            with st.spinner("Generating meeting summary with GPT-4o Mini..."):
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
                                if not os.environ.get("CLAUDE_API_KEY"):
                                    st.warning("Claude API key is required. Please enter it in the sidebar.")
                                else:
                                    claude_meeting_summary = generate_claude_summary(transcription, "meeting")
                                    if claude_meeting_summary:
                                        st.markdown("### Meeting Summary (Claude 3.7)")
                                        # Apply formatting to the Claude meeting summary
                                        formatted_claude_summary = format_meeting_summary(claude_meeting_summary)
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
                                if not os.environ.get("CLAUDE_API_KEY"):
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
                                if not os.environ.get("CLAUDE_API_KEY"):
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
    st.markdown("""
    This app uses OpenAI's speech recognition technology to transcribe and translate audio.
    
    ### Available Models:
    - **Whisper-1**: OpenAI's standard speech recognition model (for file upload)
      - Supports both transcription and translation to English
      - Handles large files (>500MB) by automatically splitting into smaller chunks
      - Memory-efficient processing for very large audio files
    - **GPT-4o Transcribe**: Latest model with real-time capabilities
    - **GPT-4o Transcribe Mini**: Lighter and faster real-time transcription model
    - **Claude 3 Sonnet**: Used for generating alternative AI summaries when the Claude option is selected
    
    ### Features:
    - Record audio via microphone
    - Real-time streaming transcription with WebSockets
    - Upload audio files for transcription
    - Translate non-English audio to English text
    - Process large audio files by automatically splitting them
    - Generate meeting summaries with multiple AI models
    
    ### Technical Details:
    The real-time transcription uses OpenAI's Realtime API with WebSockets to process audio streams as you speak, providing instant transcription using the GPT-4o Transcribe model.
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and OpenAI's speech recognition technology")

# Add CSS for summary container and category styling
st.markdown("""
<style>
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

/* Enhanced styling for evaluation container and elements */
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

/* For dark mode support */
@media (prefers-color-scheme: dark) {
    .summary-container {
        background-color: #1A237E;
        color: #FFFFFF;
        border-left: 4px solid #0D1A5E;
    }
    .category-todo, .category-overall {
        color: #333333;  /* Keep dark text on light background */
    }
    .evaluation-container {
        background-color: #5E35B1;
        color: #FFFFFF;
        border-left: 4px solid #4527A0;
    }
}
</style>
""", unsafe_allow_html=True)
