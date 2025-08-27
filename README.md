# My Work Time

The AI tool for generating summaries, transcriptions, and code

## Overview

My Work Time is a Flask-based web application that leverages Natural Language Processing, Large Language Models, and Computer Vision models to help users quickly extract insights from a variety of content formats. It can:
- Summarize PDF documents, videos and audios
- Summarize your code
- Transcribe audio and video
- Generate quiz questions based on the input (PDF document, video, audio)
- Extract text from images
- Generate code from prompts
- Translate code from one language to another

## Installation

To install My Work Time, fork the Github Repo and use an IDE of choice to open the applications (Visual Studio Code is the preferred IDE). 

Then, obtain a Google API key. Follow [these](https://ai.google.dev/gemini-api/docs/api-key) instructions to obtain a key. Remember to save your API key in a secure location. Go to the app.py file and replace “ENTER-KEY” with your Gemini API key.

For video/audio files to properly be uploaded to the website you need to download FFMPEG. Here are the [instructions](https://www.gyan.dev/ffmpeg/builds/).

Finally, run the app.py file to launch the website. Use the website link in your IDE’s terminal to launch the website in your web browser. You are all set to use the website!

## Usage

[Video Summarization](https://youtu.be/vX7rOkG8S8I?si=LXFRjuIgSG6Ydt8U)

From the home page we have different buttons that take us to various modules. One for summarization, one for making quiz questions, one for video transcription and one for coding applications.

<ins>For summarization</ins>: Click the summarize button which will take you to the summarization module. Then click either the “PDF Document” button or the “Video/Audio” button based on the type of the input you want a summary of.

<ins>For transcription</ins>: Click the Transcriber button which will take you to the transcription module. Then upload the audio or video file to obtain a transcription of the input.

<ins>For quiz generation</ins>: Click the Quiz Generator button which will take you to the quiz generation module. Then click either the “PDF Document” button or the “Video/Audio” button based on the type of the input you want a quiz for.

<ins>For image transcription</ins>:  Click the Text Extractor button which will take you to the image transcription module. Then upload the image file to obtain a transcription of the input.

## About the Team

The team consists of: Luit Deka, Michael Chen, and Arush Khare. We are students at Carnegie Mellon University (CMU) in Pittsburgh, Pennsylvania. We created this project as part of the CMU AI club. Luit is an Artificial Intelligence major, Arush is a Computer Science major, and Michael is a BCSA major learning Computer Science and Music.	

## Acknowledgements

 * Hugging Face Transformers
 * Flask framework
 * Google’s Gemini 2.0 API

