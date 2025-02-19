Run All Methods Pipeline (Colab Version)
This script runs multiple ASR/Alignment methods on a batch of .wav files by:

Mounting Google Drive in Colab to access audio data and store results.
Pulling method scripts from a GitHub repository (e.g., 6_ASR_Methods_and_Models).
Iterating through each method script and applying it to the audio files.
Tracking progress in a JSON file (progress.json) to skip already-processed files.
Logging status in audio_processing.log.
Requirements
Google Colab with GPU enabled (for faster Whisper or other model inference).
A GitHub repository hosting your method scripts.
Installed Python packages:
whisper, pydub, torch, openai, pyannote.audio, langdetect, ffmpeg, azure-cognitiveservices-speech, etc.
(Optional) GitHub Token if your repo is private.
Script Workflow
Setup and Installation

Installs needed Python packages (whisper, torch, etc.).
Mounts Google Drive to /content/drive so audio files and outputs can be stored.
Clones or references your GitHub repo to fetch scripts.
Configuration

AUDIO_FOLDER: Path in Google Drive (or another location) containing .wav files.
OUTPUT_BASE: Where results are written (e.g., /content/drive/MyDrive/Output).
GITHUB_TOKEN and GITHUB_API_URL: For fetching method scripts if the repo is private.
REPROCESS: If True, the script ignores previous progress and reruns everything.
Method Scripts

Update the methods list with the paths to each Method’s script, e.g. "Method1/...py", "Method2/...py".
The script will dynamically download each .py from GitHub, inject the correct input/output paths, and run it.
Running

Once you’ve set the variables (AUDIO_FOLDER, OUTPUT_BASE, etc.) and installed dependencies, run main() inside Colab.
It will find all .wav files and process them in small batches (default batch size is 5).
Progress Tracking

A progress.json file is created in OUTPUT_BASE. It stores which files have been processed.
If REPROCESS = False, the script skips any file already in progress.json.
Logging

The script writes detailed logs to audio_processing.log, recording success or errors for each file/method.
Outputs

For each audio file (e.g. filename.wav), it creates a subfolder under OUTPUT_BASE/filename/.
Each method’s results are placed in a method-specific folder, e.g. OUTPUT_BASE/filename/Method4/.
Usage Steps in Colab
Open a New Colab Notebook
Copy or upload this script into a cell (or import it from your Drive).
Install dependencies (e.g., !pip install torch whisper pydub ...).
Mount Drive: drive.mount('/content/drive').
Set AUDIO_FOLDER and OUTPUT_BASE to your Drive paths.
Edit methods if needed to list all the method scripts you want to run.
Set REPROCESS (optional).
Run main(): The script will process .wav files, fetch scripts from GitHub, and generate outputs & logs.
Example Notebook Snippet
python

# In your Colab cell:
!git clone https://github.com/username/YourRepo.git  # If you need to clone a repo
%cd /content/drive/MyDrive/
!pip install whisper pydub torch langdetect ...
drive.mount('/content/drive', force_remount=True)

# Configure paths:
AUDIO_FOLDER = "/content/drive/MyDrive/YourAudioFolder"
OUTPUT_BASE = "/content/drive/MyDrive/Output"
REPROCESS = True

# Then run:
main()
Troubleshooting
Private Repo: Make sure GITHUB_TOKEN is set if your method scripts are private.
No .wav Files Found: Verify AUDIO_FOLDER points to your actual files.
Progress File Issues: If a file fails or you want to rerun everything, set REPROCESS = True to remove progress.json.