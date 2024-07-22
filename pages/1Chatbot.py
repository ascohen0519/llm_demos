import time
import os
import joblib
import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

st.title("AI Chatbot")

# Saves prompt and response to message history for future model input and chat window display.
def append_message(role, content):
    st.session_state.messages.append({'role': role, 'parts': [content]})

# Returns model response from message history including most recent prompt, and temperature setting.
def GiveResponse(message_history, temp):
  return st.session_state.model.generate_content(
      contents=message_history,
      generation_config=genai.GenerationConfig(temperature=temp),
      safety_settings={
          HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
          HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
      }
  )

def ReturnTemperature(creativity):
  return d_creativity[creativity]

# If not valid API key entered, direct user to initial Demo Overview page.
if 'can_run' not in st.session_state:
    st.session_state['can_run'] = False

if not st.session_state['can_run']:
  st.markdown('Please input a valid API key on the Demo Overview page')
  st.stop()

# User selections.
col1, col2 = st.columns(2)

ai_avatar = col1.selectbox('What would you like your AI\'s avatar to be?',
                              ['', '🔆','🌟','✨','🌈','🚀','🎉','🏆','🎱'],
                              help='Please select a character to use for your models responses')

user_avatar = col2.selectbox('What would you like your avatar to be?',
                                ['', '🐀','🐁','🐋','🐉','🐊','🐉','🐈','🐇','🐌','🐍','🐎','🐑','🐒','🐓','🐔','🐕','🐟','🐞','🐝','🐜','🐞','🐝','🐜','🐛','🐚','🐙','🐘','🐗','🐖','🐠','🐡','🐢','🐣','🐤','🐥','🐦','🐧','🐨','🐩','🐳','🐲','🐱','🐰','🐯','🐮','🐭','🐬','🐫','🐪','🐴','🐵','🐶','🐷','🐸','🐹','🐺','🐻','🐼'],
                                help='Please select a character to use for your responses')

ai_name = col1.text_input('What would you like your AI\'s name to be?', '')

ai_temp = col2.selectbox('How creative would you like your AI to be?',
                                  ['', 'Very creative', 'Moderately creative', 'Not creative at all'],
                                  help="""
                                  This selection dictates how predicatable (not creative) or random (very creative)
                                  the response(s) from your AI will be.
                                  """)
d_creativity = {'Very creative': 0.9, 'Moderately creative': .5, 'Not creative at all': 0.1}

d_avatar = {'model': ai_avatar, 'user': user_avatar}

# If all choices selected, initiate model and open chat window.
if ai_avatar != '' and user_avatar != '' and ai_name != '' and ai_temp != '':

    st.write('### Chat with AI✨ ')

    # If first run, instantiate model and create empty list to append future prompts for ongoing context.
    if 'ran' not in st.session_state:
        st.session_state.ran = False

    if not st.session_state.ran:
        # Create message history
        st.session_state.messages = []

        # Configure and instantiate model as general purpose chatbot, with user input temperature setting.
        genai.configure(api_key=st.session_state['gemini_api_key'])
        st.session_state.model = genai.GenerativeModel('gemini-1.5-flash',
                                                       system_instruction='You are a general purpose chatbot.')

        # Display welcome message
        intro = 'Hello, my name is %s, how may I help you today?' % ai_name
        append_message('model', intro)
        st.session_state.ran = True

    # For all messages in history, re-write.
    for message in st.session_state.messages:
        with st.chat_message(name=message['role'], avatar=d_avatar[message['role']]):
            st.markdown(message['parts'][0])

    # Once new prompt is sent, display user prompt, append to message history, and display model response in stream.
    if prompt := st.chat_input('Your message here...'):

        # Display user prompt, and append to message history.
        with st.chat_message('user', avatar=user_avatar):
            st.markdown(prompt)

        append_message('user', prompt)

        # Feed model entire history up to latest prompt, generate response with temperature setting from user input.
        response = GiveResponse(st.session_state.messages, d_creativity[ai_temp])

        # Display model response in stream.
        with st.chat_message('model', avatar=ai_avatar):
            message_placeholder = st.empty()
            full_response = ''
            for chunk in response:
                for ch in chunk.text.split(' '):
                    full_response += ch + ' '
                    time.sleep(0.05)
                    message_placeholder.write(full_response + '▌')
            message_placeholder.write(full_response)

        # Append model response to history.
        append_message('model', response.text)