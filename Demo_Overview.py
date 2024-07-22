import streamlit as st
import google.generativeai as genai
import time

st.title('💬 LLM App Demos, by Aaron Cohen')
st.caption('[source code](https://github.com/ascohen0519/llm_demos/tree/main/pages)')

'### Instructions'
'''
In order to run the demos, you'll need to input your API key:
'''
if 'gemini_api_key' not in st.session_state:
    st.session_state['gemini_api_key'] = False

st.session_state['gemini_api_key'] = st.text_input('Gemini API Key')

if not st.session_state['gemini_api_key']:
    st.info('''
    Please add your Gemini API Key to Continue.
    If you don\'t have one, you can get an API key [here](https://aistudio.google.com/app/apikey).
    ''')
    st.stop()
else:
    genai.configure(api_key=st.session_state['gemini_api_key'])
    model = genai.GenerativeModel('gemini-1.5-flash')

try:
    st.markdown(model.generate_content('write: Valid API key, thank you.').text)
    st.session_state['can_run'] = True
    #time.sleep(2)
except:
    ('This is not a valid API key, please try again')
    st.stop()

'#### Please choose a demo from the sidebar on the left:'

'## Chatbot'

'''
Chat with a bot about a topic of interest. The bot will remember your conversation from the start and can 
interact with you as with a normal conversation. Select your avatars and start chatting. 
'''

'## Document Summarizer'

'''
Upload PDFs or manually copy and paste a body of text to be summarized. You can set parameters
for the summary to be generated, including level of creativity, length and format (bullets, paragraph).
'''

'## Document Q&A'
'#### (testing in progress)'

'''
Upload PDFs or manually copy and paste a body of text. After a brief period, you'll be able to ask questions
about your upload and receive answers. You can also set (advanced) parameters for this, including chunk
size, chunk overlap, and how many potential answer results you would like to see.
'''
