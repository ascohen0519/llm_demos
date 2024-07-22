import streamlit as st
import google.generativeai as genai
import pdfplumber
from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer
import pandas as pd
import time
import numpy as np

st.title("Document Q&A")

def click_button(button):
  st.session_state[button] = True

def update_question(question):
  st.session_state.question = question

def GetEmbedding(text):
  return genai.embed_content(model='models/text-embedding-004',
                             content=text,
                             task_type="retrieval_document")["embedding"]

def FindBestResponse(query, dataframe):
  query_embedding = genai.embed_content(model='models/text-embedding-004',
                                        content=query,
                                        task_type="retrieval_query")
  dot_products = np.dot(np.stack(dataframe['embedding']), query_embedding["embedding"])
  idx = np.argmax(dot_products)
  return dataframe.iloc[idx]['text']

if 'can_run' not in st.session_state:
    st.session_state['can_run'] = False

if not st.session_state['can_run']:
  st.markdown('Please input a valid API key on the Demo Overview page')
  st.stop()

genai.configure(api_key=st.session_state['gemini_api_key'])

st.write('''
            ### Upload Text
            Upload your text via pdf or manual entry
            ''')

document_type = st.selectbox("What type of document would you like to ask questions about?",
              ['', 'Manually input text', 'PDF'],
              help="""You can either upload a PDF document, or manually copy & paste text for your Q&A""")

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

    if 'qa_choice' not in st.session_state:
        st.session_state.qa_choice = False

    if 'embedded' not in st.session_state:
        st.session_state.embedded = False

    st.button("Begin Q&A", on_click=click_button, args=('qa_choice',))

    if st.session_state.qa_choice:

        if not st.session_state.embedded:

            with st.spinner('Wait for it...'):
                time.sleep(2)

            max_tokens = 50
            tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
            splitter = TextSplitter.from_huggingface_tokenizer(tokenizer, max_tokens)
            document_chunks = []
            st.write('...splitting document into chunks')
            for page in range(len(document)):
                page_chunk = splitter.chunks(document[page])
                for i in page_chunk:
                    document_chunks.append(i)

            if not 'df' in st.session_state:
                st.session_state['df'] = pd.DataFrame()
            st.session_state.df = pd.DataFrame(document_chunks, columns=['text'])
            st.write('...getting text embeddings (this may take a few minutes)')
            st.session_state.df['embedding'] = st.session_state.df.apply(lambda x: GetEmbedding(x.text), axis=1)
            st.session_state.embedded = True

        if 'question' not in st.session_state:
            st.session_state.question = ''

        st.write('ready!')

        question = st.text_input(
            'Ask me a question about the contents of your document!',
            on_change=update_question,
            args=('question',))

        if st.session_state.question != '':
            with st.spinner('Wait for it...'):
                time.sleep(2)

            st.markdown(FindBestResponse(question, st.session_state.df))