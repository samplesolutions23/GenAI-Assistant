import apps 
import streamlit as st

st.set_page_config(
        page_title="GenAI Assistant",page_icon='🧊'
)
st.title(":blue[GenAI Assistant:] _Summarize and Chat with YouTube Videos, Blog Posts, or PDFs_")


if "messages" not in st.session_state:
    st.session_state.messages = []

if "input_url" not in st.session_state:
    st.session_state.input_url = ""

if "llm_chain" not in st.session_state:
    st.session_state["llm_chain"] = apps.bedrock_chain()

def submit():
    st.session_state.input_url = st.session_state.user_input
    st.session_state.user_input = ""


with st.container():
    inputs, pdfs = st.tabs(["Enter URL", "Upload pdf"])
    with inputs:
        st.text_input(":Gray[Enter URL of a blog post or YouTube video to generate summary ]",placeholder="https://www.youtube.com/watch?v=RfvL_423a-I", key="user_input", on_change=submit)
        input_url = st.session_state.input_url
    with pdfs:
        upload_file = st.file_uploader("Upload pdf files [1 MB Max] to generate summary!", type="pdf")
    st.caption("Using Amazon Bedrock and Anthropic's Foundation model: _anthropic.claude-3-sonnet-20240229-v1:0_")

if upload_file:
    if 'upload_input' not in st.session_state:
        st.session_state.upload_input ='yes'
        if 'response' in st.session_state:
            del st.session_state.response
        if 'prompt' in st.session_state:
            del st.session_state.prompt
        if 'v_id' in st.session_state:
            del st.session_state.v_id
        st.session_state.messages = []
        if 't_title' in st.session_state:
            del st.session_state.t_title
        try:
            result_s3 = apps.upload_to_s3(upload_file)
        except:
            st.error('Unable to upload file to S3')
            st.stop()
        if result_s3 !='err':
            llm_chain = st.session_state["llm_chain"]
            with st.spinner('Generating Summary...'):
                try:
                    response = apps.prepare_chain(llm_chain,result_s3)
                except Exception as e:
                    #st.error(e)
                    st.error('Error reported while generating summary')
                    st.stop()
                st.session_state.response= response['response']
        else:
            st.write('Unable to process the file. Plese verify Amazon S3 bucket name and/or file uploaded is correct. Refresh the page before retry.')
    del upload_file
    

if 'cnt' not in st.session_state:
    st.session_state.cnt=0

#Create sidebar to record browser  history
st.sidebar.header("Session History")
def record_history():
    urls=st.session_state.cnt
    with st.sidebar:
        while urls > 0:
            st.write(st.session_state[urls])
            urls -=1

if st.session_state.cnt >0 and st.session_state.input_url=="":
    record_history()

def generate_summary(url):
    try:
        response,v_id,t_title = apps.generate_summary(url,llm_chain)

    except Exception as e:
        st.error('Unable to retrive details for given URL, please refresh & try again.')
        st.error(e)
        st.stop()
    st.session_state.response=response["response"]
    st.session_state.v_id=v_id
    st.session_state.t_title=t_title
    return response,v_id,t_title


#Process new URL when submitted 
if input_url:
    try:
        apps.clear_memory(st.session_state["llm_chain"])
        del st.session_state.messages
        y_id = apps.get_video_id(input_url)
        if y_id:
            y_transcript = apps.fetch_youtube_transcript(y_id)
            if not y_transcript:
                st.error("video doesn't have any english transcript")
                del st.session_state.input_url

            llm_chain = st.session_state["llm_chain"]
            if 'response' in st.session_state:
                del st.session_state.response
            if 'v_id' in st.session_state:
                del st.session_state.v_id


        #Calling Bedrock API for summary generation
        with st.spinner("Working..."):
            response,v_id,t_title = generate_summary(input_url)

        #creating new history record
        st.session_state.cnt+= 1
        st.session_state[st.session_state.cnt] = input_url    
        record_history()

        #removing for next rerun
        del st.session_state.input_url
    except Exception as e :
        st.text_area('ERROR:',e)

#if page is reload during chat
if 'response' in st.session_state:
    if 't_title' in st.session_state and st.session_state.t_title !='None':
        st.subheader(st.session_state.t_title)
    else:
        st.subheader('Summary')
    if 'v_id' in st.session_state and st.session_state.v_id != 'no_video':
        st.image('https://img.youtube.com/vi/'+st.session_state.v_id+'/0.jpg')
    st.info(st.session_state.response)


if 'response' in st.session_state:
    audio_file = apps.create_speech(st.session_state.response)
    if audio_file !='None':
        st.audio(audio_file, format="audio/mpeg")
    prompt = st.chat_input("Ask me for more details!")


if 'response' in st.session_state and prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Let me think..."):
        q_ans = apps.prepare_chain(st.session_state['llm_chain'],prompt)
    st.session_state.messages.append({"role": "assistant", "content": q_ans['response']})
# Display chat messages from history on app rerun
if "messages" in st.session_state:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
