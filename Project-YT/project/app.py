
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
import nltk
nltk.download('punkt')
# define a variable to hold you app
app = Flask(__name__)
CORS(app)


# Home page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Summary page
@app.route('/summarize', methods=['POST'])
def summarize():
    # Get the video URL from the form input
    video_url = request.form['video_url']

    # Process the video URL and generate the summary (implement your own logic here)
    summary = GetUrl(video_url)
    
    # Render the summary page with the generated summary
    return render_template('index.html', summary=summary)


@app.route('/time', methods=['GET'])
def get_time():
    return str(datetime.datetime.now())



#@app.route('/api/summarize', methods=['GET'])
def GetUrl(video_url):
    """
    Called as /api/summarize?youtube_url='url'
    """
    # if user sends payload to variable name, get it. Else empty string
    #video_url = request.args.get('youtube_url', '')
    #video_url ="https://youtu.be/TGLYcYCm2FM"
    # if(len(video_url) == 0) or (not '=' in video_url):
    #   print("f")
    #   abort(404)

    response = GetTranscript(video_url)
    
    return response

def get_video_id(video_url):
    if "youtube.com" in video_url:
        # Extract the video ID from the URL
        video_id = video_url.split("v=")[1]
        # Check if there are additional query parameters
        if "&" in video_id:
            video_id = video_id.split("&")[0]
        return video_id
    elif "youtu.be" in video_url:
        # Extract the video ID from the short URL format
        video_id = video_url.split("/")[-1]
        return video_id
    else:
        return None


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
    # Create a parser object using the `PlaintextParser` class, which takes a string `text` as input.
    # The `Tokenizer` class is used to tokenize the text into individual sentences.
    # `LANGUAGE` is the language used for tokenization.
    stemmer = Stemmer(LANGUAGE)
    # Create a stemmer object using the `Stemmer` class.
    # The stemmer is used to reduce words to their base or root form.
    # `LANGUAGE` is the language for stemming.
    summarizer = Summarizer(stemmer)
    # Create a summarizer object using the `Summarizer` class, which takes a stemmer as input.
    # The summarizer is used to generate summaries from the parsed text.
    summarizer.stop_words = get_stop_words(LANGUAGE)
    # The `get_stop_words` function is used to retrieve a list of stop words for the given language.
    s = ""
    # Iterate over the sentences returned by the summarizer.
    # `SENTENCES_COUNT` specifies the desired number of sentences in the summary.
    for sentence in summarizer(parser.document, SENTENCES_COUNT):
         # Append each sentence to the `s` string by converting it to a string representation.
        s += (str)(sentence)
    # At this point, the `s` string will contain the generated summary of a segment.
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
        
        video_id = get_video_id(video_url)
        
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        duration = max(30, transcript[-1]['start'] // 5)
        # Calculate the duration of each summary segment.
        # It is determined by the maximum value between 30 and the time of the last transcript segment divided by 5.

        i, end, st = 0, 0, 0
        text, ps_text = "", ""
        summary_content = []
        while(i < len(transcript)):
            if(end - st < duration):
                 # If the duration of the current segment is less than the desired duration:
                end = transcript[i]['start'] + transcript[i]['duration']
                ps_text += transcript[i]['text']
                # Concatenate the text of the current transcript segment to the temporary summary text.
                ps_text += ". "
            else:
                # text += "[ " + StringTime(st) + " - " + StringTime(end) + "] " + SumySummarize(ps_text) + "\n\n"
                summary_content.append({"start": StringTime(
                    st), "end": StringTime(end), "text": SumySummarize(ps_text)})
                text+=SumySummarize(ps_text)
                # Create a summary content entry with the start time, end time, and summarized text of the previous segment.
                st = end
                end = transcript[i]['start'] + transcript[i]['duration']
                ps_text = transcript[i]['text']
                 # Reset the start time and end time for the next summary segment,
                # and update the temporary summary text with the current transcript segment.

            i += 1
        summary_content.append({"start": StringTime(
            st), "end": StringTime(end), "text": SumySummarize(ps_text)})
        # text += "[ " + StringTime(st) + " - " + StringTime(end) + "] " + SumySummarize(ps_text) + "\n\n"
        text+=SumySummarize(ps_text)
        # Create a summary content entry for the final segment, using the remaining start time, end time, and summarized text.
        return text
    except Exception as e:
        # GetAudio(video_url)
        # text = GetTextFromAudio()
        # print('The text is: ', text)
        # If any exception occurs during the process, return an error entry with start time, end time, and the error message.
        return [{"start": StringTime(0), "end": StringTime(0), "text": str(e)}]



# server the app when this file is run
if __name__ == '__main__':
    app.run()