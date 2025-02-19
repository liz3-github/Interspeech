# Run All Methods Pipeline (Colab Version)

This script runs multiple ASR/Alignment methods on a batch of `.wav` files by:

- **Mounting Google Drive** in Colab to access audio data and store results.
- **Pulling method scripts** from a GitHub repository (e.g., `6_ASR_Methods_and_Models`).
- **Iterating through each method script** and applying it to the audio files.
- **Tracking progress** in a JSON file (`progress.json`) to skip already-processed files.
- **Logging status** in `audio_processing.log`.

## Requirements

- **Google Colab** with GPU enabled (for faster Whisper or other model inference).
- **A GitHub repository** hosting your method scripts.
- **Installed Python packages**:
  - `whisper`, `pydub`, `torch`, `openai`, `pyannote.audio`, `langdetect`, `ffmpeg`, `azure-cognitiveservices-speech`, etc.
- **(Optional) GitHub Token** if your repository is private.

---

## Script Workflow

### Setup and Installation

- Installs required Python packages (`whisper`, `torch`, etc.).
- Mounts Google Drive to `/content/drive` to store audio files and outputs.
- Clones or references your GitHub repo to fetch method scripts.

### Configuration

Set the following variables before running the script:

- `AUDIO_FOLDER`: Path in Google Drive (or another location) containing `.wav` files.
- `OUTPUT_BASE`: Where results are written (e.g., `/content/drive/MyDrive/Output`).
- `GITHUB_TOKEN` and `GITHUB_API_URL`: For fetching method scripts if the repository is private.
- `REPROCESS`: If `True`, the script ignores previous progress and reruns everything.

### Method Scripts

- Update the **methods list** with the paths to each Method’s script, e.g.:
  ```python
  methods = ["Method1/script.py", "Method2/script.py"]
  ```
- The script dynamically downloads each `.py` from GitHub, injects the correct input/output paths, and runs it.

### Running the Script

Once you’ve set up the variables (`AUDIO_FOLDER`, `OUTPUT_BASE`, etc.) and installed dependencies:

- Run `main()` inside Colab.
- The script will find all `.wav` files and process them in small batches (default batch size: **5**).

---

## Progress Tracking

- A `progress.json` file is created in `OUTPUT_BASE`.
- It stores which files have been processed.
- If `REPROCESS = False`, the script **skips** any file already listed in `progress.json`.

### Logging

- The script writes detailed logs to `audio_processing.log`.
- Logs include **success** or **errors** for each file/method.

### Outputs

For each audio file (e.g., `filename.wav`), a subfolder is created under `OUTPUT_BASE/filename/`.
Each method’s results are placed in a method-specific folder, e.g.,

```
OUTPUT_BASE/filename/Method4/
```

---

## Usage Steps in Colab

1. **Open a New Colab Notebook**
2. **Copy or upload this script** into a cell (or import it from your Drive).
3. **Install dependencies**:
    ```sh
    !pip install torch whisper pydub langdetect
    ```
4. **Mount Google Drive**:
    ```python
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    ```
5. **Set paths**:
    ```python
    AUDIO_FOLDER = "/content/drive/MyDrive/YourAudioFolder"
    OUTPUT_BASE = "/content/drive/MyDrive/Output"
    REPROCESS = True
    ```
6. **Run the script**:
    ```python
    main()
    ```

---

## Example Notebook Snippet

In your Colab cell:

```sh
!git clone https://github.com/username/YourRepo.git  # If you need to clone a repo
%cd /content/drive/MyDrive/
!pip install whisper pydub torch langdetect

from google.colab import drive

drive.mount('/content/drive', force_remount=True)
```

Then configure paths:

```python
AUDIO_FOLDER = "/content/drive/MyDrive/YourAudioFolder"
OUTPUT_BASE = "/content/drive/MyDrive/Output"
REPROCESS = True
```

Run the script:

```python
main()
```

---

## Troubleshooting

### Private Repository Access
- Ensure `GITHUB_TOKEN` is set if your method scripts are private.

### No `.wav` Files Found
- Verify that `AUDIO_FOLDER` correctly points to your actual files.

### Progress File Issues
- If a file fails or you want to rerun everything, set `REPROCESS = True` to remove `progress.json`.

---


