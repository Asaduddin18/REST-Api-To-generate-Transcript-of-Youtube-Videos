# from flask import Flask
# import datetime
# from youtube_transcript_api import YouTubeTranscriptApi
# import json
# from transformers import pipeline
# from transformers import BartTokenizer, BartForConditionalGeneration
# import torch
# from flask import request
# from urllib.parse import urlparse
# from urllib.parse import urlparse, parse_qs
# from contextlib import suppress
# from flask import *  
# # define a variable to hold you app
# app = Flask(__name__)

# # define your resource endpoints
# @app.route('/hello')
# def index_page():
#     return "Hello world"

# @app.route('/time', methods=['GET'])
# def get_time():
#     return str(datetime.datetime.now())

# def get_yt_id(url, ignore_playlist=False):
#     # Examples:
#     # - http://youtu.be/SA2iWivDJiE
#     # - http://www.youtube.com/watch?v=_oPAwA_Udwc&feature=feedu
#     # - http://www.youtube.com/embed/SA2iWivDJiE
#     # - http://www.youtube.com/v/SA2iWivDJiE?version=3&amp;hl=en_US
#     query = urlparse(url)
#     if query.hostname == 'youtu.be': return query.path[1:]
#     if query.hostname in {'www.youtube.com', 'youtube.com', 'music.youtube.com'}:
#         if not ignore_playlist:
#         # use case: get playlist id not current video in playlist
#             with suppress(KeyError):
#                 return parse_qs(query.query)['list'][0]
#         if query.path == '/watch': return parse_qs(query.query)['v'][0]
#         if query.path[:7] == '/watch/': return query.path.split('/')[1]
#         if query.path[:7] == '/embed/': return query.path.split('/')[2]
#         if query.path[:3] == '/v/': return query.path.split('/')[2]
    
# #api/summarize?youtube_url=url
# @app.route('/api/summarize', methods=['GET'])
# def getTranscript():
    
#     url = request.args.get('youtube_url')
    
#     #url ="https://youtu.be/TGLYcYCm2FM"
   
#     video_id = get_yt_id(url)
#     # print(url)
#     # url_data = urlparse(url)
    
#     # video_id=url_data.query[2::]
    
#     transcript= str(YouTubeTranscriptApi.get_transcript(video_id))
    
#     model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
#     tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

#     # tokenize without truncation
#     inputs_no_trunc = tokenizer(transcript, max_length=None, return_tensors='pt', truncation=False)

#     # get batches of tokens corresponding to the exact model_max_length
#     chunk_start = 0
#     chunk_end = tokenizer.model_max_length  # == 1024 for Bart
#     inputs_batch_lst = []
#     while chunk_start <= len(inputs_no_trunc['input_ids'][0]):
#         inputs_batch = inputs_no_trunc['input_ids'][0][chunk_start:chunk_end]  # get batch of n tokens
#         inputs_batch = torch.unsqueeze(inputs_batch, 0)
#         inputs_batch_lst.append(inputs_batch)
#         chunk_start += tokenizer.model_max_length  # == 1024 for Bart
#         chunk_end += tokenizer.model_max_length  # == 1024 for Bart

#     # generate a summary on each batch
#     summary_ids_lst = [model.generate(inputs, num_beams=4, max_length=100, early_stopping=True) for inputs in inputs_batch_lst]

#     # decode the output and join into one string with one paragraph per summary batch
#     summary_batch_lst = []
#     for summary_id in summary_ids_lst:
#         summary_batch = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_id]
#         summary_batch_lst.append(summary_batch[0])
#     summary_all = '\n'.join(summary_batch_lst)
#     #return render_template('index.html')
#     return (summary_all)


    


# #id = 'Y8Tko2YC5hA'




# # using pipeline API for summarization task

# #***********************************************************
# # summarization = pipeline("summarization")
# # summary_text = summarization(transcript)[0]['summary_text']
# # print("Summary:", summary_text)



    


# # server the app when this file is run
# if __name__ == '__main__':
#     app.debug = True
#     app.run()

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals
from flask import Flask, jsonify
import datetime
from flask import request  # used to parse payload
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from flask import render_template
from flask import abort
from flask_cors import CORS
import os

# define a variable to hold you app
app = Flask(__name__)
CORS(app)


@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/time', methods=['GET'])
def get_time():
    return str(datetime.datetime.now())


@app.route('/api/summarize', methods=['GET'])
def GetUrl():
    """
    Called as /api/summarize?youtube_url='url'
    """
    # if user sends payload to variable name, get it. Else empty string
    video_url = request.args.get('youtube_url', '')
    # if(len(video_url) == 0) or (not '=' in video_url):
    #   print("f")
    #   abort(404)

    response = GetTranscript(video_url)
    return jsonify(response)


def SumySummarize(text):

    from sumy.parsers.html import HtmlParser
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lsa import LsaSummarizer as Summarizer
    from sumy.nlp.stemmers import Stemmer
    from sumy.utils import get_stop_words

    LANGUAGE = "english"
    SENTENCES_COUNT = 3
    import nltk

    # url = "https://en.wikipedia.org/wiki/Automatic_summarization"
    # parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
    # or for plain text files
    # parser = PlaintextParser.from_file("document.txt", Tokenizer(LANGUAGE))
    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)

    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    s = ""
    for sentence in summarizer(parser.document, SENTENCES_COUNT):
        s += (str)(sentence)
    return s


def GetTextFromAudio():
    import speech_recognition as sr
    from pydub import AudioSegment

    f = ""

    # convert mp3 file to wav
    for file in os.listdir(os.getcwd()):
        if file.endswith(".mp3"):
            f = file

    if(len(f) == 0):
        return f
    sound = AudioSegment.from_mp3(f)

    os.rename(os.path.join(os.getcwd(), f),
              os.path.join(os.getcwd(), "recordings", f))

    sound.export("transcript.wav", format="wav")

    # use the audio file as the audio source
    AUDIO_FILE = "transcript.wav"

    r = sr.Recognizer()
    with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.record(source)  # read the entire audio file
        return (r.recognize_google(audio))


def GetAudio(video_url):
    from youtube_dl import YoutubeDL
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3'
        }],
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])


def StringTime(time):
    time = (int)(time)
    return (str)(time // 60) + ":" + (str)(time % 60)

# video id are the last characters in the link of youtube video


def GetTranscript(video_url):
    text = ""
    try:
        video_id = (video_url.split('=')[1]).split("&")[0]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        duration = max(30, transcript[-1]['start'] // 5)
        i, end, st = 0, 0, 0
        text, ps_text = "", ""
        summary_content = []
        while(i < len(transcript)):
            if(end - st < duration):
                end = transcript[i]['start'] + transcript[i]['duration']
                ps_text += transcript[i]['text']
                ps_text += ". "
            else:
                # text += "[ " + StringTime(st) + " - " + StringTime(end) + "] " + SumySummarize(ps_text) + "\n\n"
                summary_content.append({"start": StringTime(
                    st), "end": StringTime(end), "text": SumySummarize(ps_text)})
                st = end
                end = transcript[i]['start'] + transcript[i]['duration']
                ps_text = transcript[i]['text']

            i += 1
        summary_content.append({"start": StringTime(
            st), "end": StringTime(end), "text": SumySummarize(ps_text)})
        # text += "[ " + StringTime(st) + " - " + StringTime(end) + "] " + SumySummarize(ps_text) + "\n\n"
        return summary_content
    except Exception as e:
        # GetAudio(video_url)
        # text = GetTextFromAudio()
        # print('The text is: ', text)
        return [{"start": StringTime(0), "end": StringTime(0), "text": str(e)}]


# server the app when this file is run
if __name__ == '__main__':
    app.run()