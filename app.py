import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
#from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub

st.set_page_config(page_title='🤗💬 PDF Chat App - GPT')

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 PDF Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model

    ''')
  #  add_vertical_space(5)
    st.write('Made with ❤️ by [NOA COHEN]')




def main():
    st.header("Talk to your PDF 💬")
    st.write("This app uses OpenAI's LLM model to answer questions about your PDF file. Upload your PDF file and ask questions about it. The app will return the answer from your PDF file.")

    st.header("1. Pass your HUGGINGFACE API KEY here")
    huggingfacehub_api_token = st.text_input("**HUGGINGFACE API KEY**")
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingfacehub_api_token
    #v='demo'
 
   # huggingfacehub_api_token=st.text_input("**HUGGINGFACE API KEY**", value=v)
   # st.write("You can get your OpenAI API key from [here](https://platform.openai.com/account/api-keys)")


  #  if huggingfacehub_api_token==v:
   #     huggingfacehub_api_token=st.secrets["HUGGINGFACE_API_KEY"]
    # if huggingfacehub_api_token=='':
    #     load_dotenv()
    #os.environ["HUGGINGFACE_API_KEY"] = huggingfacehub_api_token

    # upload a PDF file

    st.header("2. Upload PDF")
    pdf = st.file_uploader("**Upload your PDF**", type='pdf')

    # st.write(pdf)

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)

        # # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
        # st.write(chunks)

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')s
        else:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                )
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        

    # st.header("or.. Try it with this The Alchaemist PDF book")
    # if st.button('Ask The Alchemist Book Questions'):
    #     with open("The_Alchemist.pkl", "rb") as f:
    #         VectorStore = pickle.load(f)
        # Accept user questions/query
        st.header("3. Ask questions about your PDF file:")
       # q="Tell me about the content of the PDF"
        query = st.text_input("Questions") #value=q
        # st.write(query)

        if st.button("Ask"):
            if huggingfacehub_api_token == '':
                st.write('Warning: Please pass your OPEN AI API KEY on Step 1')
           # st.write(huggingfacehub_api_token)
            #os.environ["HUGGINGFACE_API_KEY"] = huggingfacehub_api_token
            #if huggingfacehub_api_token=='':
             #   st.write('Warning: Please pass your HUGGINGFACE API KEY on Step 1')
            else:
                docs = VectorStore.similarity_search(query=query, k=3)

                llm = HuggingFaceHub(repo_id="google/flan-t5-xxl",
                    model_kwargs={"temperature":0.5, "max_length":512},)
                  #  huggingfacehub_api_token='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')


                chain = load_qa_chain(llm=llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=query)
                    print(cb)
                st.header("Answer:")
                st.write(response)
                st.write('--')
                st.header("HUGGINGFACE API Usage:")
                st.text(cb)

if __name__ == '__main__':
    main()
