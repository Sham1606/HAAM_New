import numpy as np
import sounddevice as sd
import queue
import logging
import threading
import time
from .utils import AUDIO_config, get_logger

logger = get_logger("AudioStream")

class AudioStreamManager:
    def __init__(self, callback_function):
        """
        callback_function: called when a complete turn is detected. 
                           Signature: func(audio_data: np.ndarray, timestamp: float)
        """
        self.callback = callback_function
        self.q = queue.Queue()
        self.running = False
        self.stream = None
        
        # VAD Parameters
        self.sample_rate = AUDIO_config["SAMPLE_RATE"]
        self.block_size = AUDIO_config["BLOCK_SIZE"]
        self.threshold = AUDIO_config["VAD_THRESHOLD_DB"]
        self.silence_dur = AUDIO_config["SILENCE_DURATION"]
        
        # State
        self.buffer = []
        self.is_speaking = False
        self.silence_start = None
        self.turn_start_time = None
        
        # Pre-calculate threshold in linear scale
        # DB = 20 * log10(RMS) -> RMS = 10^(DB/20)
        self.rms_threshold = 10 ** (self.threshold / 20)

    def _audio_callback(self, indata, frames, time, status):
        """Called by sounddevice for each audio block"""
        if status:
            logger.warning(f"Audio status: {status}")
        self.q.put(indata.copy())

    def _process_loop(self):
        """Main processing loop running in a separate thread"""
        while self.running:
            try:
                # Get block from queue
                data = self.q.get(timeout=0.5)
                # Ensure mono
                if data.ndim > 1:
                    data = data.mean(axis=1)
                
                # Calculate energy (RMS)
                rms = np.sqrt(np.mean(data**2))
                
                # VAD Logic
                if rms > self.rms_threshold:
                    # Speech detected
                    if not self.is_speaking:
                        self.is_speaking = True
                        self.turn_start_time = time.time()
                        logger.debug("Speech started")
                    
                    self.silence_start = None # Reset silence timer
                    self.buffer.append(data)
                
                else:
                    # Silence detected
                    if self.is_speaking:
                        self.buffer.append(data) # Keep appending for a bit
                        
                        if self.silence_start is None:
                            self.silence_start = time.time()
                        
                        # Check if silence duration exceeded
                        elif time.time() - self.silence_start > self.silence_dur:
                            # End of turn
                            self._flush_turn()
                            
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in audio processing: {e}")

    def _flush_turn(self):
        """Finalize turn and send to callback"""
        if not self.buffer:
            return
            
        full_audio = np.concatenate(self.buffer)
        duration = len(full_audio) / self.sample_rate
        
        if duration >= AUDIO_config["MIN_TURN_DURATION"]:
            logger.info(f"Turn detected: {duration:.2f}s")
            # Run callback in a separate thread to not block audio processing
            threading.Thread(target=self.callback, args=(full_audio, self.turn_start_time)).start()
        
        # Reset state
        self.buffer = []
        self.is_speaking = False
        self.silence_start = None

    def start(self):
        if self.running:
            return
        
        self.running = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=1,
            callback=self._audio_callback
        )
        self.stream.start()
        
        self.thread = threading.Thread(target=self._process_loop)
        self.thread.start()
        logger.info("Microphone stream started. Listening...")

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        logger.info("Microphone stream stopped.")
        
if __name__ == "__main__":
    # Simple test
    def debug_cb(audio, ts):
        print(f"Captured {len(audio)} samples at {ts}")
        
    manager = AudioStreamManager(debug_cb)
    try:
        manager.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        manager.stop()
