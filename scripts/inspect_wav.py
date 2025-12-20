
import os
import struct

path = r"d:\haam_framework\crema-d-mirror-main\AudioWAV\1001_DFA_ANG_XX.wav"
print(f"File size: {os.path.getsize(path)}")
with open(path, 'rb') as f:
    header = f.read(44)
    print(f"Header: {header}")
    try:
        import wave
        with wave.open(path, 'rb') as w:
            print(f"Channels: {w.getnchannels()}")
            print(f"Width: {w.getsampwidth()}")
            print(f"Framerate: {w.getframerate()}")
            print(f"Frames: {w.getnframes()}")
            print("Wave module opened successfully")
    except Exception as e:
        print(f"Wave module failed: {e}")
