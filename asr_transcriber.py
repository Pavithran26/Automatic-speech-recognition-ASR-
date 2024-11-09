import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
import librosa

model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

def transcribe_audio(file_path):
    audio_input, sample_rate = librosa.load(file_path, sr=16000)
    
    inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    
    return transcription[0]

if __name__ == "__main__":
    audio_file_path = "sample.wav"  
    transcription_result = transcribe_audio(audio_file_path)
    print("Transcription:", transcription_result)
