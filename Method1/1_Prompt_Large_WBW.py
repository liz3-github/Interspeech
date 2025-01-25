import os
import whisper
import json
import re
from google.colab import files

# 设置音频文件路径
AUDIO_FILE = "{input_file}"
OUTPUT_FOLDER = "{output_folder}"

def transcribe_audio(file_path):
    print("加载Whisper large模型...")
    model = whisper.load_model("large")

    # 设置whisper转录选项
    whisper_options = {
        "verbose": None,
        "word_timestamps": True,
        "task": "transcribe",
        "suppress_tokens": "",
        "language": None,  # 自动检测语言
        "temperature": 0.0,  # 降低随机性
        "condition_on_previous_text": True  # 考虑上下文
    }

    print("开始转录...")

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
            words_with_timestamps.append({
                "word": word['word'],
                "start": (word['start']),
                "end": (word['end'])
            })

        formatted_results.append({
            "Timestamp": f"{segment['start']} - {segment['end']}",
            "Words": words_with_timestamps,
            "Lang": language
        })
    return formatted_results

def main():
    try:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        # 转录
        transcription = transcribe_audio(AUDIO_FILE)

        if transcription:
            # 格式化结果
            final_results = process_and_format_transcription(transcription)

            # 保存结果
            transcription_file = os.path.join(OUTPUT_FOLDER, "Method1_large_WBW.json")
            with open(transcription_file, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, ensure_ascii=False, indent=2)

            print(f"转录结果已保存至 {transcription_file}")
        else:
            print("转录失败。")

    except Exception as e:
        print(f"处理过程中发生错误: {e}")

if __name__ == "__main__":
    main()