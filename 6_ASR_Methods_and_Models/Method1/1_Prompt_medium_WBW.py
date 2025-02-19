import os
import whisper
import json
import re
from google.colab import files

# Set audio file path and output folder
AUDIO_FILE = "{input_file}"
OUTPUT_FOLDER = "{output_folder}"

def format_timestamp(seconds: float) -> str:
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

def transcribe_audio(file_path):
    print("Loading Whisper small model...")
    model = whisper.load_model("medium")

    # Set transcription options
    whisper_options = {
        "verbose": None,
        "word_timestamps": True,
        "task": "transcribe",
        "suppress_tokens": "",

    }

    print("Starting transcription...")

    result = model.transcribe(
        file_path,
        **whisper_options
    )

    return result


def process_and_format_transcription(result):
    formatted_results = []
    language = result['language']

    for segment in result['segments']:
        words_with_timestamps = []
        for word in segment['words']:
            w_start_str = format_timestamp(word["start"])
            w_end_str = format_timestamp(word["end"])

            words_with_timestamps.append({
                "word": word['word'],
                "start": w_start_str,
                "end": w_end_str
            })

        formatted_results.append({
            "Timestamp": f"{segment['start']} - {segment['end']}",
            "Words": words_with_timestamps,
            "Language": language
        })
    return formatted_results

def main():
    try:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        # Transcribe audio
        transcription = transcribe_audio(AUDIO_FILE)

        if transcription:
            # Format the transcription results
            final_results = process_and_format_transcription(transcription)

            # Save the results to a JSON file
            transcription_file = os.path.join(OUTPUT_FOLDER, "Method1_medium_WBW.json")
            with open(transcription_file, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, ensure_ascii=False, indent=2)

            print(f"Transcription saved to {transcription_file}")
        else:
            print("Transcription failed.")

    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main()
