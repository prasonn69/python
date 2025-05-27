from pinecone import Pinecone
import fitz
import google.generativeai as genai
import os
PINECONE_API_KEY='pcsk_2bD2JV_KYSPf2H3NaGbKEKmUKsF3XTVx94xZ6jecUkNadDv72nC9V1fBXicXpUdgtJfLeM'
GEMINI_API_KEY='AIzaSyBk64_SI4Bh8eL-DXc-24SR1vuOluK9REw'
def extract_text_from_pdf(pdf_path):
    text=''
    if ".pdf" in pdf_path:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text+=page.get_text()+'\n'
    return text
model='models/text-embedding-004'
genai.configure(api_key=GEMINI_API_KEY)
def embed_text(text):
    response=genai.embed_content(
        model=model,
        content=text,
        task_type='retrieval_document',
    )
    return response
pinecone_client=Pinecone(api_key=PINECONE_API_KEY)
vector_index=pinecone_client.Index('studentkb')
def upsert_vectors_to_pinecone(document_text):
    upsert_data=[]
    for idx, (file,text) in enumerate(document_text.items()):
        vector=embed_text(text).get('embedding',[])
        meta_data={
            'text': text,
        }
        upsert_data.append((f'doc={idx}',vector,meta_data))
    vector_index.upsert(upsert_data)
    print('Vectors upserted successfully')


if __name__=='__main__':
    document_text={}

    for file in os.listdir('documents'):
        text=extract_text_from_pdf('documents/'+file)
        if text:
            document_text[file]=text
    upsert_vectors_to_pinecone(document_text)
    print('All documents processed and vectors upserted.')


