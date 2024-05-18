import os
from azure.storage.blob import BlobServiceClient
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores.azuresearch import AzureSearch
import streamlit as st
from tqdm import tqdm
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter


# Define Azure Storage connection details
connection_string = "DefaultEndpointsProtocol=https;AccountName=ajtsdevsistrg;AccountKey=oaY0INq+3IRBDzuD5tO1Av9YVloi8fs3CHecExkrPZ4r6PRRaThCKL3BZ2vL4V5+w+bqp/Ai8NVT+AStTBOY3g==;EndpointSuffix=core.windows.net"
container_name = "genai"

# Initialize Azure Blob Storage connection
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)

# Initialize Azure Search vector store details
vector_store_address = "https://ai-search-ajuserv.search.windows.net"
vector_store_password = "9ODMu6tpu6y3B6nYzVd9bTabY8qVLL0FWnWRTxpJwrAzSeCFUbYa"
index_name = "langchain-vector-demo"

# Initialize OpenAI details
openai_api_key = "sk-proj-6xlSz9blKZhweCZO7ERDT3BlbkFJNW3HopOfPKkmBrhnvWxY"
openai_api_version = "2023-05-15"
model = "text-embedding-ada-002"

# Initialize LangChain components
embeddings = OpenAIEmbeddings(
    openai_api_key=openai_api_key,
    openai_api_version=openai_api_version,
    model=model
)

vector_store = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
)

text_splitter = CharacterTextSplitter(chunk_size=256, chunk_overlap=20)
# docs = text_splitter.split_documents(document)
 
# vector_store.add_documents(documents=docs)

llm_open = OpenAI(api_key=openai_api_key, temperature=0.6)
qa = RetrievalQA.from_chain_type(llm=llm_open, chain_type="stuff", retriever=vector_store.as_retriever(), return_source_documents=False)

# Function to transcribe audio files from Azure Blob Storage and get bot response
def transcribe_and_get_bot_response():
    # Get all blobs with the desired extension
    blobs = container_client.list_blobs()
    num_files = sum(1 for blob in blobs if blob.name.endswith(('.m4a', '.mpeg', '.mp3')))
    
    with tqdm(total=num_files, desc="Transcribing Files") as pbar:
        for blob in blobs:
            if blob.name.endswith(('.m4a', '.mpeg', '.mp3')):
                filename = os.path.basename(blob.name)

                try:
                    # Download the blob data to a temporary location
                    os.makedirs("temp", exist_ok=True)
                    download_path = f"temp/{filename}"
                    with open(download_path, "wb") as audio_file:
                        blob_client = container_client.get_blob_client(blob.name)
                        audio_data = blob_client.download_blob().readall()
                        audio_file.write(audio_data)

                    # Generate transcription
                    # Placeholder transcription (replace with actual transcription code)
                    transcription = f"Transcription for {blob.name}"

                    # Get bot response based on transcription
                    bot_response = qa.run(query=transcription)

                    # Display transcription and bot response
                    st.write(f"Transcription for {blob.name}: {transcription}")
                    st.write("Chatterbean:", bot_response)
                    
                    pbar.update(1)

                except Exception as e:
                    st.error(f"An exception occurred at file {blob.name}: {str(e)}")

# Main function for Streamlit app
def main():
    st.title("CabCritique: Shaping Service Excellence")
    
    # Button to trigger transcription and bot response
    if st.button("Transcribe and Get Bot Response"):
        # transcribe_and_get_bot_response()
        pass
    

    # Input area for user query
    user_query = st.text_input("Privileged to assist with your queries")
    
    # Button to trigger the bot's response
    if st.button("Ask"):
        # Get bot response based on user query
        bot_response = qa.run(query=user_query)
        
        # Display bot's response
        st.write("Chatterbean:", bot_response)

if __name__ == "__main__":
    main()
