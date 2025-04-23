"""
Unit test for audio file splitting functionality
"""

import os
import tempfile
from pydub import AudioSegment
import numpy as np
import math

# Import the split_audio_file function from app.py
# We need to add the current directory to the Python path
import sys
sys.path.append('.')
from app import split_audio_file

def create_test_audio_file(duration_ms=60000, format='mp3', file_size_mb=30):
    """Create a test audio file with the specified duration and target file size"""
    # Create a simple sine wave
    sample_rate = 44100
    t = np.linspace(0, duration_ms/1000, int(sample_rate * duration_ms/1000))
    audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    # Convert to 16-bit PCM
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Create an AudioSegment
    audio = AudioSegment(
        audio_data.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,  # 16-bit
        channels=1
    )
    
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{format}')
    temp_path = temp_file.name
    temp_file.close()
    
    # Export to the specified format with a bitrate that will approximate the target file size
    # This is an approximation - exact file size control is difficult
    target_bitrate = int((file_size_mb * 8 * 1024) / (duration_ms / 1000))
    audio.export(temp_path, format=format, bitrate=f"{target_bitrate}k")
    
    # Check the actual file size
    actual_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
    print(f"Created test audio file: {temp_path}")
    print(f"Target size: {file_size_mb}MB, Actual size: {actual_size_mb:.2f}MB")
    
    return temp_path, actual_size_mb

def test_split_audio_file():
    """Test that a large MP3 file is correctly split into smaller chunks"""
    # Create a test MP3 file that's larger than the limit
    test_file_path, actual_size = create_test_audio_file(
        duration_ms=180000,  # 3 minutes
        format='mp3',
        file_size_mb=30  # Target size of 30MB, similar to 09Apr25.mp3
    )
    
    try:
        # Split the file
        chunk_paths, was_split = split_audio_file(test_file_path)
        
        # Verify the file was split
        if not was_split:
            print("ERROR: The file should have been split")
            return False
        
        if len(chunk_paths) <= 1:
            print("ERROR: The file should have been split into multiple chunks")
            return False
        
        # Verify each chunk is under the limit
        all_chunks_valid = True
        for i, chunk_path in enumerate(chunk_paths):
            size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
            print(f"Chunk {i+1} size: {size_mb:.2f}MB")
            
            if size_mb > 25:
                print(f"ERROR: Chunk {i+1} exceeds the 25MB limit: {size_mb:.2f}MB")
                all_chunks_valid = False
            
            # Also verify the chunk is in MP3 format
            extension = os.path.splitext(chunk_path)[1].lower()
            if extension != '.mp3':
                print(f"ERROR: Chunk should have .mp3 extension, got {extension}")
                all_chunks_valid = False
            
            # Clean up the chunk file
            os.unlink(chunk_path)
        
        if all_chunks_valid:
            print(f"Successfully split {actual_size:.2f}MB file into {len(chunk_paths)} chunks")
            print(f"All chunks are below the 25MB limit and maintain the MP3 format")
            return True
        else:
            return False
    
    finally:
        # Clean up the test file
        if os.path.exists(test_file_path):
            os.unlink(test_file_path)

if __name__ == "__main__":
    # Run the test directly
    success = test_split_audio_file()
    if success:
        print("Test PASSED!")
    else:
        print("Test FAILED!") 