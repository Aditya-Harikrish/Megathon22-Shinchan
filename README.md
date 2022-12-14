# Megathon-22 Shinchan

## What is this?
This repository contains a Python program to take an image as input, perform OCR to extract text from it, detect any particular language in it (or not), translate the text into English (or any other language of your choice), and replace the original text in the image with the translated text as well as use TTS to read it out loud for the user in the target language.

<p align="center">
    <img src="img/10.jpeg" alt="Original Image" width="400"/>
    <img src="img/10_output.jpg" alt="Original Image" width="400"/>
<p>

For more details, please refer to the [full presentation](https://github.com/Aditya-Harikrish/Megathon22-Shinchan/blob/main/A_case_for_EVs_to_make_travels_more_accessible_removing_the_language_barrier_compressed.pdf)

## Setup

For the entire setup, please refer to Google Cloud's documentation of setting up Google Cloud Vision in Python. Download your API key, and name it `key.json` in the Megathon22-Shinchan directory. 

Some of the commands you may need to run are:
```sh
pip install google-cloud-vision     # for OCR
pip install googletrans==3.1.0a0    # for translation
pip install gTTS playsound=1.2.2    # for TTS and playing audio
pip install opencv-contrib-python   # for image processing
```

To run (with Python3):
```
python main.py
```

## Team Shinchan
- [@Aditya-Harikrish](https://github.com/Aditya-Harikrish)
- [@VKohli17](https://github.com/VKohli17)
- [@hardikgu23](https://github.com/hardikgu23)
- [@Vidhi1109](https://github.com/Vidhi1109)