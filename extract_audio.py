#!/usr/bin/env python3
import argparse
import os
from moviepy.editor import VideoFileClip

def extract_audio(video_path, output_path=None):
    """
    Extract audio from a video file and save it as an MP3.
    
    Args:
        video_path (str): Path to the input video file
        output_path (str, optional): Path to save the output MP3 file. 
                                    If not provided, will use the same name as the video file.
    
    Returns:
        str: Path to the saved MP3 file
    """
    try:
        # Validate that the input file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # If output path is not specified, use the input filename with .mp3 extension
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = f"{base_name}.mp3"
        
        print(f"Extracting audio from {video_path}...")
        # Load the video file
        video_clip = VideoFileClip(video_path)
        
        # Extract the audio
        audio_clip = video_clip.audio
        
        # Save the audio as MP3
        audio_clip.write_audiofile(output_path)
        
        # Close the clips to free up resources
        audio_clip.close()
        video_clip.close()
        
        print(f"Audio extraction complete. Saved to: {output_path}")
        return output_path
    
    except Exception as e:
        print(f"Error extracting audio: {str(e)}")
        return None

def main():
    """Parse arguments and extract audio from video."""
    parser = argparse.ArgumentParser(description="Extract audio from video file as MP3")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("-o", "--output", help="Path to save the output MP3 file (optional)")
    
    args = parser.parse_args()
    
    extract_audio(args.video_path, args.output)

if __name__ == "__main__":
    main() 