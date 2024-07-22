import streamlit as st
import pdfplumber
import time
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# User input to determine temperature setting of summarization model
d_creativity = {'Very creative': 1, 'Moderately creative': .5, 'Not creative at all': 0}

# Prompt template for summarization.
summary_prompt = """The following is a document.
DOCUMENT: %s
INSTRUCTIONS: Please write your summary in %s format. %s.
SUMMARY:
"""

def click_button(button):
  st.session_state[button] = True

def ReturnTemperature(creativity):
  return d_creativity[creativity]

def GiveResponse(prompt, temp):
  return model.generate_content(
    contents=prompt,
    generation_config=genai.GenerationConfig(temperature=temp),
    safety_settings={
      HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
      HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
    }
  ).text

def SummarizeDocument(document, creativity, bullets_or_summary, max_bullets, max_length):

  temp = ReturnTemperature(creativity)

  paragraph_limit = 'the paragraph should be approximately %s characters' % max_length
  bullets_limit = 'please include %s bullets' %max_bullets


  if bullets_or_summary == 'Bullets':
    summary_format = 'bullet'
    length_limit = bullets_limit
  elif bullets_or_summary == 'Paragraph':
    summary_format = 'a single paragraph'
    length_limit = paragraph_limit
  elif bullets_or_summary == 'Bullets & Paragraph':
    summary_format = 'a single paragraph and bullets'
    length_limit = paragraph_limit + ', and ' + bullets_limit
  else:
    return 'error'

  text_to_summarize = ['. '.join(document)][0]

  prompt = summary_prompt%(text_to_summarize, summary_format, length_limit)

  return GiveResponse(prompt, temp)

st.title("Document Summarizer")

if 'can_run' not in st.session_state:
    st.session_state['can_run'] = False

if not st.session_state['can_run']:
  st.markdown('Please input a valid API key on the Demo Overview page')
  st.stop()

genai.configure(api_key=st.session_state['gemini_api_key'])
model = genai.GenerativeModel('gemini-1.5-flash', system_instruction='You are a document summarizer.')

st.write('''
            ### Upload Text
            Upload your text via pdf or manual entry
            ''')

document_type = st.selectbox("What type of document would you like to summarize?",
              ['', 'Manually input text', 'PDF'],
              help="""You can either upload a PDF document, or manually copy & paste text to summarize""")

if document_type != '':
  document = []

  if document_type == 'PDF':
    uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
    if uploaded_file:
      st.success("Uploaded the file")
      with pdfplumber.open(uploaded_file) as file:
        all_pages = file.pages
        for i in all_pages:
          document.append(i.extract_text())
  else:
    out = st.text_area('Enter your text, then click Upload', '', height=200)

    if 'text_entered' not in st.session_state:
      st.session_state.text_entered = False

    st.button("Upload", on_click=click_button, args=('text_entered', ))

    if st.session_state.text_entered:
      with st.spinner('Wait for it...'):
                time.sleep(2)
      document.append(out)


  if len(document) > 0:
    st.write('The length of your document is %s characters.'%len(document[0]))

    st.write('''
                ### Set Summarization Parameters
                Decide the format, length and creativity level for your summary. 
                ''')

    bullets_or_summary = st.selectbox("What format would you like the summary in?",
                                      ['', 'Bullets', 'Paragraph', 'Bullets & Paragraph'])

    if 'summarize' not in st.session_state:
      st.session_state.summarize = False

    if bullets_or_summary != '':

      max_bullets, max_length = 0, 0

      if 'Bullets' in bullets_or_summary:
        max_bullets = st.select_slider("How many bullets for the summary?",
                                       options=[''] + list(range(2, 11)))

      if 'Paragraph' in bullets_or_summary:
        max_length = st.select_slider("How many characters for the summary?",
                                      options=[''] + list(range(50, 2001, 50)))

      if (
              (bullets_or_summary == 'Bullets' and max_bullets != '') or
              (bullets_or_summary == 'Paragraph' and max_length != '') or
              (bullets_or_summary == 'Bullets & Paragraph' and max_bullets != '' and max_length != '')):

        creativity = st.selectbox("How creative would you like the summary to be?",
                                  ['', 'Very creative', 'Moderately creative', 'Not creative at all'])

        if creativity != '':
          st.write("### Generate Summary")
          st.markdown('Click Summarize to Continue...')
          st.button("Summarize", on_click=click_button, args=('summarize',))

          if st.session_state.summarize:
            with st.spinner('Wait for it...'):
              time.sleep(5)

            try:
              st.markdown(SummarizeDocument(document, creativity, bullets_or_summary, max_bullets, max_length))
              st.session_state.summarize = False
              st.write('\nThank you for trying this demo. Have a nice day!')
            except:
              st.write('Not a valid document type, please try again')