from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
import requests
import argparse
import csv


parser = argparse.ArgumentParser(description='AI LLM parser.')
parser.add_argument('--question', type=str, help='Ask a question or send a command to LLM')
args = parser.parse_args()

llm = Ollama(model="mistral")

# loader = JSONLoader(
#     file_path='devices.json',
#     jq_schema='.[]',
#     text_content=False)

# docs = loader.load()

# def format_docs(docs):
#     print("start doc list:")
#     print(docs)
#     print("end doc list")
#     doc_count = len(docs)
#     print(f"Total docs found: {doc_count}")
#     return "\n\n".join(doc.page_content for doc in docs)

# embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# documents = loader.load()
# db = Chroma.from_documents(documents, embedding_function)

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={'score_threshold': 0.1}
# )

# Function to load and filter CSV content based on a keyword
def load_and_filter_csv(file_path, keyword):
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        rows = [row for row in reader if keyword.lower() in str(row).lower()]
    return rows

# Function to load and format CSV content
def load_csv(file_path):
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    return rows

def format_csv(rows):
    formatted_rows = "\n".join([", ".join(row.values()) for row in rows])
    return formatted_rows

# Load CSV content
csv_rows = load_csv('data.csv')

# Use LLM to determine relevant keywords or criteria
criteria_prompt = PromptTemplate(
    input_variables=['question'],
    template="Extract the most relevant criteria from the following question to filter CSV content: \nQuestion: {question} \nCriteria:"
)

criteria_chain = (
    {"question": args.question}
    | criteria_prompt
    | llm
    | StrOutputParser()
)

criteria = criteria_chain.invoke(f"{args.question}").strip()


# Filter CSV content based on the keyword
filtered_csv_content = format_csv(load_and_filter_csv('data.csv', criteria))

main_prompt = PromptTemplate(
    input_variables=['question', 'context'],
    template="You are an assistant for determining AWS resource costs. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. \nQuestion: {question} \nContext: {context} \nAnswer:"
)

rag_chain = (
    #{"context": retriever | format_docs, "question": RunnablePassthrough()}
    #| prompt
    #{"context": {}, "question": RunnablePassthrough()}
    #{"context": csv_content, "question": RunnablePassthrough()}
    {"context": filtered_csv_content, "question": RunnablePassthrough()}
    | main_prompt
    | llm
    | StrOutputParser()
)

result = rag_chain.invoke(f"{args.question}")
print(result)