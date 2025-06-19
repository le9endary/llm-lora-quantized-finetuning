import os
import librosa
import soundfile as sf
import numpy as np
import webrtcvad
from openai import OpenAI

api_key = ''
client = OpenAI(api_key=api_key)

def correct_transcription(transcription):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for the company Omantel. Your task is to correct any spelling discrepancies or grammar error in the transcribed text."},
            {"role": "user", "content": f"Correct the following Arabic transcription in Arabic. If there is no mistake, return the same transcription. Do not write any other word beside transcription in your output: {transcription}"}
        ]
    )
    corrected_transcription = response.choices[0].message.content.strip()
    return corrected_transcription

def remove_silence(input_file, top_db=20):
    y, sr = librosa.load(input_file, sr=None)
    non_silent_intervals = librosa.effects.split(y, top_db=top_db)
    y_trimmed = np.concatenate([y[start:end] for start, end in non_silent_intervals])
    return y_trimmed, sr

def apply_vad(y, sr, frame_duration_ms=30):
    vad = webrtcvad.Vad()
    vad.set_mode(1)
    y = (y * 32767).astype(np.int16)
    frame_length = int(sr * frame_duration_ms / 1000.0)
    voiced_frames = []
    for start in range(0, len(y), frame_length):
        end = min(start + frame_length, len(y))
        frame = y[start:end]
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)), 'constant')
        is_speech = vad.is_speech(frame.tobytes(), sr)
        if is_speech:
            voiced_frames.append(frame)
    y_vad = np.concatenate(voiced_frames)
    y_vad = y_vad.astype(np.float32) / 32767
    return y_vad

def segment_audio(y, sr, segment_duration=10, overlap_duration=2):
    segment_length = int(segment_duration * sr)
    overlap_length = int(overlap_duration * sr)
    segments = []
    start = 0
    while start < len(y):
        end = start + segment_length
        segment = y[start:end]
        if len(segment) < segment_length:
            segment = np.pad(segment, (0, segment_length - len(segment)), 'constant')
        segments.append(segment)
        start += segment_length - overlap_length
    return segments

def transcribe_audio_segment(segment, sr, segment_path):
    sf.write(segment_path, segment, sr)
    try:
        with open(segment_path, 'rb') as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json"
            )
        return transcription
    except Exception as e:
        print(f"Error transcribing {segment_path}: {e}")
        return None

def process_audio_files(input_folder, output_folder, original_transcript_folder, corrected_transcript_folder, segment_duration=10, overlap_duration=2):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(original_transcript_folder):
        os.makedirs(original_transcript_folder)
    if not os.path.exists(corrected_transcript_folder):
        os.makedirs(corrected_transcript_folder)

    files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]
    total_files = len(files)

    for processed_files, filename in enumerate(files, start=1):
        input_path = os.path.join(input_folder, filename)
        output_basename = os.path.splitext(filename)[0]

        # Remove silence
        y_trimmed, sr = remove_silence(input_path)

        # Apply VAD
        y_vad = apply_vad(y_trimmed, sr)

        # Segment audio
        segments = segment_audio(y_vad, sr, segment_duration, overlap_duration)

        # Process each segment
        for i, segment in enumerate(segments):
            segment_filename = f"{output_basename}_segment_{i + 1}.wav"
            segment_path = os.path.join(output_folder, segment_filename)

            transcription = transcribe_audio_segment(segment, sr, segment_path)

            if transcription and transcription.language == 'arabic':
                original_transcription = transcription.text
                corrected_transcription = correct_transcription(original_transcription)

                original_transcript_filename = f"{output_basename}_segment_{i + 1}.txt"
                original_transcript_path = os.path.join(original_transcript_folder, original_transcript_filename)
                with open(original_transcript_path, 'w', encoding='utf-8') as transcript_file:
                    transcript_file.write(original_transcription)

                corrected_transcript_filename = f"{output_basename}_segment_{i + 1}.txt"
                corrected_transcript_path = os.path.join(corrected_transcript_folder, corrected_transcript_filename)
                with open(corrected_transcript_path, 'w', encoding='utf-8') as transcript_file:
                    transcript_file.write(corrected_transcription)

                print(f"Processed and transcribed segment {i + 1} of {filename}")
            else:
                os.remove(segment_path)
                print(f"Removed segment {i + 1} of {filename} due to non-Arabic transcription")

        print(f"Processed {processed_files} of {total_files} files. {total_files - processed_files} files left.")

input_folder = 'D:/python/final_whisper_datacreation/set4/728wav'
output_folder = 'D:/python/final_whisper_datacreation/set4/output'
original_transcript_folder = 'D:/python/final_whisper_datacreation/set4/original_transcripts'
corrected_transcript_folder = 'D:/python/final_whisper_datacreation/set4/corrected_transcripts'
process_audio_files(input_folder, output_folder, original_transcript_folder, corrected_transcript_folder)