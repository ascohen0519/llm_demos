import time
import os
import joblib
import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

st.title('AI Chatbot')


def append_message(role, content):
    """Appends message to message history to pass into model during each new prompt in session.

    Args:
        role: Model or user.
        content: Content in message (completion from model or prompt from user).
    """
    st.session_state.messages.append({'role': role, 'parts': [content]})


def GiveResponse(message_history, gen_config):
    """Generates model response given message history including latest user prompt.

    Args:
        message_history: Session state message history, containing all prior messages and latest user prompt.
        gen_config: Dictionary of top-p and temperature setting based on user input creativity level.

    Returns:
        Model response, includes .text for displaying the prompt completion in UI.
    """
    return st.session_state.model.generate_content(
        contents=message_history,
        generation_config=genai.GenerationConfig(**gen_config),
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
        }
    )


# If not valid API key entered, direct user to initial Demo Overview page.
if 'can_run' not in st.session_state:
    st.session_state['can_run'] = False

if not st.session_state['can_run']:
    st.markdown('Please input a valid API key on the Demo Overview page')
    st.stop()

# User selections.
col1, col2 = st.columns(2)

ai_avatar = col1.selectbox('What would you like your AI\'s avatar to be?',
                           ['', '🔆', '🌟', '✨', '🌈', '🚀', '🎉', '🏆', '🎱'])

user_avatar = col2.selectbox('What would you like your avatar to be?',
                             ['', '🐀', '🐁', '🐋', '🐉', '🐊', '🐉', '🐈', '🐇', '🐌', '🐍', '🐎', '🐑', '🐒', '🐓', '🐔', '🐕', '🐟',
                              '🐞', '🐝', '🐜', '🐞', '🐝', '🐜', '🐛', '🐚', '🐙', '🐘', '🐗', '🐖', '🐠', '🐡', '🐢', '🐣', '🐤', '🐥',
                              '🐦', '🐧', '🐨', '🐩', '🐳', '🐲', '🐱', '🐰', '🐯', '🐮', '🐭', '🐬', '🐫', '🐪', '🐴', '🐵', '🐶', '🐷',
                              '🐸', '🐹', '🐺', '🐻', '🐼'])

ai_name = col1.text_input('What would you like your AI\'s name to be?', '')

ai_creativity = col2.selectbox('How creative would you like your AI to be?',
                               ['', 'Not creative at all', 'Moderately creative', 'Very creative'],
                               help='''
                                      This decides how your model chooses the next word in it's response to you - by
                                      altering the number of tokens in consideration (top-p) and the shape of the
                                      probability distribution (temperature) for token selection.
                                      ''')

d_avatar = {'model': ai_avatar, 'user': user_avatar}

# If all choices selected, initiate model and open chat window.
if ai_avatar != '' and user_avatar != '' and ai_name != '' and ai_creativity != '':

    d_creativity_temp = {'Not creative at all': 0, 'Moderately creative': 1, 'Very creative': 2}
    d_creativity_topp = {'Not creative at all': 0, 'Moderately creative': .5, 'Very creative': 1}
    gen_config = {'temperature': d_creativity_temp[ai_creativity], 'top_p': d_creativity_topp[ai_creativity]}

    st.write('### Chat with AI✨ ')

    # If first run, instantiate model and create empty list to append future prompts for ongoing context.
    if 'ran' not in st.session_state:
        st.session_state.ran = False

    if not st.session_state.ran:
        # Create message history.
        st.session_state.messages = []

        # Configure and instantiate model as general purpose chatbot, with user input temperature setting.
        genai.configure(api_key=st.session_state['gemini_api_key'])
        st.session_state.model = genai.GenerativeModel(
            'gemini-1.5-flash',
            system_instruction='You are a general purpose chatbot. Your name is %s' % ai_name)

        # Display welcome message
        intro = 'Hello, my name is %s, how may I help you today?' % ai_name
        append_message('model', intro)
        st.session_state.ran = True

    # For all messages in history, re-write in chat ui.
    for message in st.session_state.messages:
        with st.chat_message(name=message['role'], avatar=d_avatar[message['role']]):
            st.markdown(message['parts'][0])

    # Once new prompt is sent, display user prompt, append to message history, and display model response in stream.
    if prompt := st.chat_input('Your message here...'):

        # Display user prompt, and append to message history.
        with st.chat_message('user', avatar=user_avatar):
            st.markdown(prompt)

        # Append latest user prompt to history.
        append_message('user', prompt)

        # Feed model entire history up to latest prompt, gen_config set based on user input creativity level.
        response = GiveResponse(st.session_state.messages, gen_config)

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
