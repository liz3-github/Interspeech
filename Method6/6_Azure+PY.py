import azure.cognitiveservices.speech as speechsdk
import time
import json
from pyannote.audio import Pipeline
import torch
import os
import re

def diarize_audio(hf_auth_token, audio_file):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_auth_token)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(torch.device(device))
    diarization = pipeline(audio_file)
    return diarization

# 把 Display 里的标点对齐插入到 words_array
def merge_punctuation(words_array, display_text):
    """
    1) 用正则把 display_text 拆成 '单词' 或 '标点' token
    2) 顺序匹配到 words_array 里, 遇到标点就贴到上一个单词末尾
    """
    # 把所有"连续字母数字"或"非字母数字的符号"分别提取成token
    tokens = re.findall(r"\w+|[^\w\s]", display_text)
    # 例如 "Hello world, nice!" -> ["Hello", "world", ",", "nice", "!"]

    final_words = []
    w_index = 0

    for token in tokens:
        # 如果 token 全是非字母数字(标点符号), 就贴到上一个单词
        if re.match(r"^\W+$", token) and final_words:
            final_words[-1]["word"] += token
        else:
            # 如果是正常单词, 去对齐 words_array
            if w_index < len(words_array):
                # 复制 words_array[w_index], 只改 'word'
                item = dict(words_array[w_index])
                item["word"] = token
                final_words.append(item)
                w_index += 1
            else:
                # 如果 tokens 比 words_array 多, 这里就忽略
                pass

    return final_words

def main():
    speech_key, service_region = "6ZpjiQJliTBcaUYuOtoYtFNrWo4uxrTTeK6CGNElsV9HKlqnB9XiJQQJ99BAACYeBjFXJ3w3AAAYACOGA8D8", "eastus"
    AUDIO_FILE = "{input_file}"
    OUTPUT_FOLDER = "{output_folder}"   
    weatherfilename = AUDIO_FILE
    hf_auth_token = "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"  # 替换为你的Hugging Face token

    # 确保输出文件夹存在
    output_folder = OUTPUT_FOLDER
    os.makedirs(output_folder, exist_ok=True)

    # 进行说话者分离
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
            timestamp_str = f"{offset_seconds:.2f} - {(offset_seconds + duration_seconds):.2f}"

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
                            "start": w_start,
                            "end": w_end,
                            "probability": w.get("Confidence", 0.0)
                        })

            # === 新增 ===>
            # 把display_text里的标点分给words_array
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
            # 打印带标点的整句
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
        # 将结果写入JSON文件
        with open(os.path.join(output_folder, "Method6_Model_Azure_WBW.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("Transcription End. Results saved to JSON file.")

if __name__ == "__main__":
    main()