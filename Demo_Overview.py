import streamlit as st
import google.generativeai as genai

st.title('💬 LLM App Demos, by Aaron Cohen')
st.caption('[source code](https://github.com/ascohen0519/llm_demos/tree/main/pages)')

'### Instructions'
'''
In order to run the demos, you'll need to input your Google Cloud API key to run Gemini:
'''

# Prompt user to input API key.
if 'gemini_api_key' not in st.session_state:
    st.session_state['gemini_api_key'] = False

st.session_state['gemini_api_key'] = st.text_input('Gemini API Key', type='password')

if not st.session_state['gemini_api_key']:
    st.info('''
    Please add your API Key to Continue.
    If you don\'t have one, you can create a new API key [here](https://aistudio.google.com/app/apikey).
    ''')
    st.stop()
else:
    genai.configure(api_key=st.session_state['gemini_api_key'])
    model = genai.GenerativeModel('gemini-1.5-flash')

# Test valid API key through successful model completion.
try:
    model.generate_content('hello')
    st.info('''
        Valid API key. Thank you.
        ''')
    st.write('\n')
    st.session_state['can_run'] = True
except:
    st.info('''
        This is not a valid API key. Please try again.
        ''')
    st.stop()

'#### Please choose a demo from the sidebar on the left:'

'## RAG Document Q&A'

'''
Upload a PDF or manually enter a body of text to ask questions about. After a brief period, you'll be able to ask
questions and receive answers. You can set parameters including document chunking method, size and overlap, as well
as text embedding model choice.
'''

'## Chatbot'

'''
Chat with an AI chatbot about a topic of interest. The chatbot will remember your conversation history throughout the
session. You can customize the chatbot's name, avatar and desired creativity level. Add your company name and logo
to personalize the UI to your company.
'''

'## Document Summarizer'

'''
Upload a PDF or manually enter a body of text to be summarized. You can set parameters for your summary including
format, length, and optional advanced parameters including token sampling method and temperature.
'''




