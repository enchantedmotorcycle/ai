import csv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
import argparse

parser = argparse.ArgumentParser(description='AI LLM parser.')
parser.add_argument('--question', type=str, help='Ask a question to LLM')
args = parser.parse_args()

llm = Ollama(model="mistral")

# Function to load and filter CSV content based on vCPU and Memory
def load_and_filter_csv(file_path, vcpu, memory):
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        rows = [row for row in reader if int(row['vCPU']) >= vcpu and float(row['Memory'].replace(' GiB', '')) >= memory]
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
csv_rows = load_csv('AWS_EC2.csv')

# Use LLM to determine relevant vCPU and Memory requirements
criteria_prompt = PromptTemplate(
    input_variables=['question'],
    template="Extract the vCPU and Memory requirements from the following question: \nQuestion: {question} \nvCPU: \nMemory:"
)

criteria_chain = (
    {"question": RunnablePassthrough()}
    | criteria_prompt
    | llm
    | StrOutputParser()
)

criteria = criteria_chain.invoke(f"{args.question}").strip().split('\n')
print(criteria)
vcpu = int(criteria[0].split(':')[1].strip())
memory = float(criteria[1].split(':')[1].strip().replace(' GiB', ''))

# Filter CSV content based on the vCPU and Memory requirements
filtered_csv_content = format_csv(load_and_filter_csv('AWS_EC2.csv', vcpu, memory))

# Create the main prompt
main_prompt = PromptTemplate(
    input_variables=['question', 'context'],
    template="""
    You are an assistant for determining AWS resource costs. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    Question: {question}
    Context: {context}
    Answer:
    """
)

rag_chain = (
    {"context": filtered_csv_content, "question": RunnablePassthrough()}
    | main_prompt
    | llm
    | StrOutputParser()
)

result = rag_chain.invoke(f"{args.question}")
print(result)