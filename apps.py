import boto3
import json
import time
import re
import os
import requests
import env
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse
from transformers import pipeline
from bs4 import BeautifulSoup
from contextlib import closing
#from streamlit_pdf_viewer import pdf_viewer
import streamlit as st
from langchain_community.chat_models import BedrockChat
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain


ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
region = os.getenv("region")
s3_bucket = env.s3_bucket_name
session = boto3.Session(region_name=region,aws_access_key_id=ACCESS_KEY,aws_secret_access_key=SECRET_KEY)

def create_beddrock_client():
        bedrock_client = session.client("bedrock-runtime")
        return bedrock_client


def create_polly_client():
    polly = session.client('polly')
    return polly

def create_textract_client():
    textract = session.client('textract')
    return textract

def create_s3_client():
    s3_client = session.client('s3')
    return s3_client

if s3_bucket == '':
    st.error('Please update the Amazon S3 bucket name. This S3 bucket will be used to store document which you upload for summary and chatting. ')
    st.stop()


def upload_to_s3(file):
    try:
        with st.spinner('Uploading file to S3...'):
            s3_client = create_s3_client()
            response_s3 = s3_client.upload_fileobj(file, s3_bucket,file.name )
        
        with st.spinner('Extracting text from uploaded file...'):
            result_pdf = pdf_text(s3_bucket,file.name)

        return result_pdf 
    except:
        return 'err'



def pdf_text(s3_bucket,s3_file):
    try:
        textract = create_textract_client()
        result = textract.start_document_text_detection(
                DocumentLocation={
                    'S3Object': {
                    'Bucket': s3_bucket,
                    'Name': s3_file
                    }})
        job_id = result['JobId']

        if is_job_complete(textract, job_id):
            response_job = get_job_results(textract, job_id)

        if response_job:
            doc =''
            for result_page in response_job:
                for item in result_page["Blocks"]:
                    if item["BlockType"] == "LINE":
                        doc += item["Text"] +'\n'

        return doc 
    except Exception as e:
        return e


def is_job_complete(client, job_id):
    time.sleep(1)
    response = client.get_document_text_detection(JobId=job_id)
    status = response["JobStatus"]
    print("Job status: {}".format(status))

    while(status == "IN_PROGRESS"):
        time.sleep(1)
        response = client.get_document_text_detection(JobId=job_id)
        status = response["JobStatus"]
        print("Job status: {}".format(status))

    return status


def get_job_results(client, job_id):
    pages = []
    time.sleep(1)
    response = client.get_document_text_detection(JobId=job_id)
    pages.append(response)
    print("Resultset page received: {}".format(len(pages)))
    next_token = None
    if 'NextToken' in response:
        next_token = response['NextToken']

    while next_token:
        time.sleep(1)
        response = client.\
            get_document_text_detection(JobId=job_id, NextToken=next_token)
        pages.append(response)
        print("Resultset page received: {}".format(len(pages)))
        next_token = None
        if 'NextToken' in response:
            next_token = response['NextToken']

    return pages


def create_speech(input_text):
    output_format = "mp3"
    voice_id = "Matthew"
    polly = create_polly_client()
    response = polly.synthesize_speech(
            Text=input_text,
            OutputFormat=output_format,
            VoiceId=voice_id
            )
    file_name='genai_assistant.mp3'
    if "AudioStream" in response:
            with closing(response["AudioStream"]) as stream:
                output = os.path.join(os.getcwd(), file_name)
                try:
                    with open(output, "wb") as file:
                        file.write(stream.read())
                        return (file_name)
                except IOError as error:
                    print(f"Error: {error}")
    else:
        print("None")

def bedrock_chain():

    try: 
        model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
        model_kwargs =  {
            "max_tokens": 2048,
            "temperature": 0.0,
            "top_k": 250,
            "top_p": 1,
            "stop_sequences": ["\n\nHuman"],
        }
        model = BedrockChat(
            client=create_beddrock_client(),
            model_id=model_id,
            model_kwargs=model_kwargs,
        )

        prompt_template = """System: I want you to provide brief summary of this input provided, and then list the key points or key takeaways. Summary should not be more than 200 words. Add conclusion at the end. you should not use keywords like "input provided" or "200 words" in your response. If you don't have answer to the input,just say it and stop.

        Current conversation:
        {history}

        User: {input}
        Assistant:"""
        prompt = PromptTemplate(
            input_variables=["history", "input"], template=prompt_template
        )

        memory = ConversationBufferMemory(human_prefix="User", ai_prefix="Assistant")
        conversation = ConversationChain(
            prompt=prompt,
            llm=model,
            verbose=True,
            memory=memory,
        )
        return conversation
    except Exception as e:
        st.error(e)
        st.stop

def clear_memory(chain):
    return chain.memory.clear()

def prepare_chain(chain, prompt):
    return chain({"input": prompt})


def create_description_embedding( text):
        payload = {
            "prompt": f"\n\nHuman:{text}\n\nAssistant:",
            "max_tokens_to_sample": 2048,
            "temperature": 0.0,
            "top_k": 250,
            "top_p": 1,
        }
        model= "anthropic.claude-instant-v1"
        body = json.dumps(payload)
        accept = "application/json"
        contentType = "application/json"

        response = bedrock_client.invoke_model(
           body=body, modelId=model, accept=accept, contentType=contentType
        )
        response_body = json.loads(response.get("body").read())

        return  response_body.get("completion", "").strip()


def lang_memory():

    memory = ConversationBufferMemory(human_prefix="User", ai_prefix="assistant")
    return memory

def lang_chain():
    conversation = ConversationChain(
        prompt=prompt,
        llm=model,
        verbose=True,
        memory=lang_memory,
    )

def fetch_blogs(url):
    try:
        # Send a GET request to the specified URL
        response = requests.get(url)

        # Raise an exception if the request was unsuccessful
        response.raise_for_status()

        # Parse the HTML content of the page with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract all text content from the page
        # You may need to adapt this to target specific tags/classes/IDs based on the page structure
        transcript = soup.get_text(separator='\n')
        title_tag = soup.find('meta', property='og:title')
        b_title = title_tag['content'] if title_tag else 'None'
        return transcript.strip(),b_title

    except requests.RequestException as e:
        return f"An error occurred while fetching the transcript from the URL: {e}"


def get_video_id(url):
    # Extract the video ID from the YouTube URL
    video_id = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    return video_id.group(1) if video_id else None

def fetch_youtube_transcript(video_id):
    try:
        # Fetch the transcript using youtube-transcript-api
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        # Combine the text from the transcript entries
        transcript = ' '.join([entry['text'] for entry in transcript_list])
        return transcript
    
    except Exception as e:
        return 'None' 

def get_y_title(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text)
    link = soup.find_all(name="title")[0]
    title = str(link)
    title = title.replace("<title>","")
    title = title.replace("</title>","")
    title = title.replace("- YouTube","")
    return title

def generate_summary(url,llm_chain):
    url_domain=urlparse(url).netloc
    if url_domain == 'www.youtube.com':
        video_id = get_video_id(url)
        if video_id:
            transcript = fetch_youtube_transcript(video_id)
            summary = prepare_chain(llm_chain,transcript)
            y_title = get_y_title(url)
            return summary,video_id,y_title
        else:
            return "Invalid YouTube URL."
    else:
        transcript,b_title = fetch_blogs(url)
        summary = prepare_chain(llm_chain,transcript)
        return summary,'no_video',b_title


