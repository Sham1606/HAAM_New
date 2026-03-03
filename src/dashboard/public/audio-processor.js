class MicAudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (input && input.length > 0) {
      const channelData = input[0];
      if (channelData) {
        // Send raw Float32Array channel data back to the main thread
        // We copy the buffer to ensure it transfers safely without detachment issues
        this.port.postMessage(new Float32Array(channelData));
      }
    }
    return true; // Keep processor alive
  }
}

registerProcessor('mic-audio-processor', MicAudioProcessor);
