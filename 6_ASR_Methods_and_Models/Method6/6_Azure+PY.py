import azure.cognitiveservices.speech as speechsdk
import time
import json
from pyannote.audio import Pipeline
import torch
import os
import re

def format_timestamp(seconds: float) -> str:
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

def diarize_audio(hf_auth_token, audio_file):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_auth_token)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(torch.device(device))
    diarization = pipeline(audio_file)
    return diarization

# Insert punctuation from the "Display" field into words_array
def merge_punctuation(words_array, display_text):
    """
    1) Use regex to split display_text into tokens of either 'words' or 'punctuation'
    2) Align the tokens sequentially with words_array; if punctuation is encountered,
       append it to the end of the previous word.
    """
    # Extract tokens: either sequences of alphanumeric characters or individual non-alphanumeric symbols
    tokens = re.findall(r"\w+|[^\w\s]", display_text)
    # For example, "Hello world, nice!" -> ["Hello", "world", ",", "nice", "!"]

    final_words = []
    w_index = 0

    for token in tokens:
        # If the token consists entirely of non-alphanumeric characters (i.e., punctuation), append it to the previous word
        if re.match(r"^\W+$", token) and final_words:
            final_words[-1]["word"] += token
        else:
            #  For a regular word, align it with words_array
            if w_index < len(words_array):
                # Copy the current entry from words_array and update only the 'word' field
                item = dict(words_array[w_index])
                item["word"] = token
                final_words.append(item)
                w_index += 1
            else:
                # If there are more tokens than words in words_array, ignore the extras
                pass

    return final_words

def main():
    speech_key, service_region = "6ZpjiQJliTBcaUYuOtoYtFNrWo4uxrTTeK6CGNElsV9HKlqnB9XiJQQJ99BAACYeBjFXJ3w3AAAYACOGA8D8", "eastus"
    AUDIO_FILE = "{input_file}"
    OUTPUT_FOLDER = "{output_folder}"   
    weatherfilename = AUDIO_FILE
    hf_auth_token = "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" 

    
    output_folder = OUTPUT_FOLDER
    os.makedirs(output_folder, exist_ok=True)

    # Perform speaker diarization
    diarization = diarize_audio(hf_auth_token, weatherfilename)

    endpoint_string = f"wss://{service_region}.stt.speech.microsoft.com/speech/universal/v2"
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, endpoint=endpoint_string)
    audio_config = speechsdk.audio.AudioConfig(filename=weatherfilename)

    speech_config.output_format = speechsdk.OutputFormat.Detailed

    speech_config.set_property(
        speechsdk.PropertyId.SpeechServiceResponse_PostProcessingOption,
        "TrueText"
    )

    speech_config.set_property(
        property_id=speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode,
        value='Continuous'
    )
    auto_detect_source_language_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
        languages=["en-US", "zh-CN"]
    )

    conversation_transcriber = speechsdk.transcription.ConversationTranscriber(
        speech_config=speech_config,
        audio_config=audio_config,
        auto_detect_source_language_config=auto_detect_source_language_config
    )
    results = []
    done = False

    def recognized_cb(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            detected_language = evt.result.properties.get(
                speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
            )

            offset_seconds = evt.result.offset / 1e7
            duration_seconds = evt.result.duration / 1e7
            timestamp_str = f"{format_timestamp(offset_seconds)} - {format_timestamp(offset_seconds + duration_seconds)}"

            speaker = "Unknown"
            for turn, _, speaker_label in diarization.itertracks(yield_label=True):
                if turn.start <= offset_seconds <= turn.end:
                    speaker = speaker_label
                    break

            detailed_json = json.loads(evt.result.json)
            words_array = []
            display_text = ""

            if "NBest" in detailed_json and len(detailed_json["NBest"]) > 0:
                best_result = detailed_json["NBest"][0]
                display_text = best_result.get("Display", "")

                if "Words" in best_result:
                    for w in best_result["Words"]:
                        w_offset_s = w["Offset"] / 1e7
                        w_duration_s = w["Duration"] / 1e7
                        w_start = w_offset_s
                        w_end = w_offset_s + w_duration_s
                        words_array.append({
                            "word": w["Word"],
                            "start": format_timestamp(w_start),
                            "end": format_timestamp(w_end),
                            "probability": w.get("Confidence", 0.0)
                        })

            
            # Distribute punctuation from display_text into words_array
            if words_array and display_text:
                words_array = merge_punctuation(words_array, display_text)

            segment_result = {
                "Timestamp": timestamp_str,
                "Speaker": speaker,
                "TextWithPunctuation": display_text,
                "Words": words_array,
                "Language": detected_language if detected_language else ""
            }
            results.append(segment_result)

            print(f"RECOGNIZED: {timestamp_str}")
            print(f"Speaker: {speaker}")
            print(f"Language: {detected_language}")
            # Print the complete sentence with punctuation
            print(f"Display Text: {display_text}")
            print(f"Words Count: {len(words_array)}\n")

    def session_stopped_cb(evt):
        print('SESSION STOPPED {}'.format(evt))
        nonlocal done
        done = True

    conversation_transcriber.transcribed.connect(recognized_cb)
    conversation_transcriber.transcribing.connect(lambda evt: print('TRANSCRIBING: {}'.format(evt)))
    conversation_transcriber.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
    conversation_transcriber.session_stopped.connect(session_stopped_cb)
    conversation_transcriber.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))

    conversation_transcriber.start_transcribing_async()

    try:
        while not done:
            time.sleep(.5)
    except KeyboardInterrupt:
        print("Stopping transcription.")
    finally:
        conversation_transcriber.stop_transcribing_async()
        
        with open(os.path.join(output_folder, "Method6_Model_Azure_WBW.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("Transcription End. Results saved to JSON file.")

if __name__ == "__main__":
    main()