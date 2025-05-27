

GEMINI_API_KEY = 'AIzaSyBk64_SI4Bh8eL-DXc-24SR1vuOluK9REw'
GROQ_API_KEY = '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'
PINECONE_API_KEY = 'pcsk_2bD2JV_KYSPf2H3NaGbKEKmUKsF3XTVx94xZ6jecUkNadDv72nC9V1fBXicXpUdgtJfLeM'

from pinecone import Pinecone
from create_vectors import embed_text
from groq import Groq
import streamlit as st

pinecone_client = Pinecone(api_key='pcsk_2bD2JV_KYSPf2H3NaGbKEKmUKsF3XTVx94xZ6jecUkNadDv72nC9V1fBXicXpUdgtJfLeM')
vector_index = pinecone_client.Index('studentkb')

groq_client = Groq(api_key=GROQ_API_KEY)

st.write('A cool RAG App Created by Broadway AI Students Batch 12')
st.write('This is a simple RAG app that uses Pinecone and groq to answer the questions based on the documents ')

query = st.text_input('Enter your question about students:')
if query:
    vector = embed_text(query).get('embedding', [])
    response = vector_index.query(vector=vector, top_k=1, include_metadata=True)
    similar_document = response['matches'][0]['metadata']['text']
    prompt = [{
        'role': 'user',
        'content': f'You are provided with a document that may contain relevant information:\n\n{similar_document}\n\nUser query: {query}.If the question is not in the document you should provide answer based in your knowledge'
    }]
    llm_response = groq_client.chat.completions.create(
        model='llama3-70b-8192',
        messages=prompt,
        max_tokens=700,
        temperature=0.7,
    )
    answer = llm_response.choices[0].message.content
    st.write(" Answer:")
    st.write(answer)