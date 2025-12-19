import torchaudio as ta

from chatterbox import ChatterboxTTS

if __name__ == '__main__':
    model = ChatterboxTTS.from_local(device="cuda", ckpt_dir="model")
    text = "chamam Sheng Mobei de 'Senhor Sheng'?"
    wav = model.generate(text)
    ta.save("test-english.wav", wav, model.sr)
