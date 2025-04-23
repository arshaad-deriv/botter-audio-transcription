# Audio Transcription App

A Streamlit application that converts speech to text using OpenAI's advanced speech recognition technology. This app allows users to transcribe audio through microphone recording or file upload, and includes real-time transcription capabilities with OpenAI's Realtime API.

![Audio Transcription App](https://i.imgur.com/placeholder.png)

## Features

- **Microphone Recording**: Record audio directly from your device's microphone
- **Real-time Transcription**: Stream audio in real-time and see the transcription as you speak using OpenAI's Realtime API with WebSockets
- **File Upload**: Upload audio files (MP3, MP4, WAV, M4A) for transcription
- **Large File Support**: Process audio files larger than 500MB with memory-efficient processing
- **Multiple Models**: Choose between OpenAI's Whisper-1 and GPT-4o Transcribe models
- **Translation**: Translate non-English audio to English text using Whisper model
- **Meeting Summaries**: Generate structured meeting summaries from transcriptions
- **User-friendly Interface**: Clean and intuitive design with progress indicators for large files

## Requirements

- Python 3.7+
- OpenAI API key (required)
- Claude API key (optional, for enhanced summaries)
- Approximately 500MB of RAM for processing large audio files

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/audio-transcription-app.git
   cd audio-transcription-app
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. For large file support, create a `.streamlit/config.toml` file:
   ```
   server.maxUploadSize=1000
   ```

5. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage

### Microphone Recording

1. Select the "Microphone Recording" tab
2. Choose your preferred recording duration and speech recognition model
3. Click "Start Recording" and speak into your microphone
4. After recording completes, the app will display the transcription

### Real-time Transcription with WebSockets

1. Select the "Real-time Transcription with GPT-4o" tab
2. Click "Start Streaming" and begin speaking
3. The transcription will appear in real-time as you speak, using OpenAI's Realtime API
4. Click "Stop Streaming" when you're done

### File Upload

1. Select the "File Upload" tab
2. Choose your preferred speech recognition model
3. For Whisper-1 model, you can optionally enable translation to English
4. For advanced summaries, you can enable Claude 3.7 summaries (requires Claude API key)
5. Upload an audio file (MP3, MP4, WAV, M4A) - supports files up to 1000MB
6. Click "Transcribe and Summarize" to process the file
7. For large files, progress bars will show the processing status
8. View the transcription and automatically generated summaries

### Meeting Summaries

The app can automatically generate structured meeting summaries with:
1. Overall meeting summary
2. Categorized breakdown (Completed, Ongoing, Blockers, Ideas, To Do)
3. Action points with owners and deadlines
4. Option to use either GPT-4o Mini or Claude 3.7 for summary generation

## Models

- **Whisper-1**: OpenAI's standard speech recognition model
  - Supports both transcription and translation to English
  - Handles large files by automatically splitting into smaller chunks
  - Has a 25MB per-request limit (app handles this automatically)
  - Ideal for accurate transcription of pre-recorded audio
- **GPT-4o Transcribe**: Latest model with real-time capabilities
  - Used for real-time streaming transcription via WebSockets
  - Higher accuracy for complex audio
  - No file size splitting required
- **GPT-4o Transcribe Mini**: Lighter and faster real-time transcription model
  - More efficient for real-time processing
  - Slightly lower accuracy than the full GPT-4o Transcribe model
- **Claude 3.7 Sonnet**: Used for generating alternative AI summaries
  - Optional for enhanced meeting summaries
  - Requires Claude API key

## Technical Details

This application uses:
- Streamlit for the web interface
- OpenAI's Realtime API with WebSockets for real-time transcription using GPT-4o Transcribe
- OpenAI's Audio API for file-based transcription using Whisper
- Python's sounddevice and soundfile libraries for audio recording and processing
- WebSocket client for real-time communication with OpenAI's API
- Threading for non-blocking audio processing
- Memory-efficient chunking for processing large audio files (>500MB)
- Progress indicators for tracking large file processing

### Large File Support

The app includes special handling for large audio files:
1. Files are automatically split into manageable chunks based on size
2. Memory-efficient processing with explicit garbage collection
3. Progress bars for loading and processing large files
4. Streamlit server configuration to allow files up to 1000MB (default is 200MB)

To enable large file uploads, create a `.streamlit/config.toml` file with:
```
server.maxUploadSize=1000
```

## License

MIT

## Acknowledgements

- [OpenAI](https://openai.com/) for their speech recognition API
- [Streamlit](https://streamlit.io/) for the web app framework
