import numpy as np
import sounddevice as sd
import queue
import logging
import threading
import time

logger = logging.getLogger(__name__)

# Audio Configuration
AUDIO_config = {
    "SAMPLE_RATE": 16000,
    "BLOCK_SIZE": 4096,        # Processing chunk size
    "VAD_THRESHOLD_DB": -35,   # Energy threshold for speech
    "SILENCE_DURATION": 1.5,   # Seconds of silence to end a turn (increased from 0.6 to 1.5 so it waits longer before sending)
    "MIN_TURN_DURATION": 2.0,  # Minimum speech duration to process (increased from 0.8 to 2.0 for better context)
    "MAX_TURN_DURATION": 15.0  # Max duration to force a cut (increased from 8.0 to 15.0)
}

class AudioStreamManager:
    def __init__(self, callback_function):
        """
        callback_function: called when a complete turn is detected. 
                           Signature: func(audio_data: np.ndarray, timestamp: float)
        """
        self.callback = callback_function
        self.q = queue.Queue()
        self.callback_q = queue.Queue() # Prevents thread explosion
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
        self.rms_threshold = 10 ** (self.threshold / 20)

    def _audio_callback(self, indata, frames, time, status):
        """Called by sounddevice for each audio block"""
        if status:
            pass # Suppressed constant warning to avoid console spam
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

    def _callback_worker_loop(self):
        """Processes turns strictly ONE AT A TIME so we don't crash Windows/CPU"""
        while self.running:
            try:
                full_audio, start_time = self.callback_q.get(timeout=0.5)
                # Only process if audio is long enough
                if len(full_audio) >= int(AUDIO_config["MIN_TURN_DURATION"] * self.sample_rate):
                    self.callback(full_audio, start_time)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Callback worker failed: {e}")

    def _flush_turn(self):
        """Finalize turn and send to callback queue"""
        if not self.buffer:
            return
            
        full_audio = np.concatenate(self.buffer)
        duration = len(full_audio) / self.sample_rate
        
        if duration >= AUDIO_config["MIN_TURN_DURATION"]:
            logger.info(f"Turn detected: {duration:.2f}s... Added to processing queue.")
            # Put in queue instead of creating a new thread each time
            try:
                self.callback_q.put((full_audio, self.turn_start_time))
            except Exception as e:
                logger.error(f"Failed to queue callback: {e}")
        
        # Reset state
        self.buffer = []
        self.is_speaking = False
        self.silence_start = None

    def start(self):
        if self.running:
            return
        
        self.running = True
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=1,
                callback=self._audio_callback
            )
            self.stream.start()
            
            # Start mic consumer
            self.thread = threading.Thread(target=self._process_loop)
            self.thread.start()
            
            # Start prediction sequencer worker
            self.worker_thread = threading.Thread(target=self._callback_worker_loop)
            self.worker_thread.start()
            
            logger.info("Microphone stream started. Listening...")
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            self.running = False

    def stop(self):
        self.running = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logger.error(f"Error stopping stream: {e}")
        logger.info("Microphone stream stopped.")
