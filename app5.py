import requests
import csv
from langchain.agents import load_tools
from langchain_community.llms import Ollama
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import JSONLoader
import json
from pathlib import Path
from pprint import pprint
import os
from langchain_core.tools import tool
import requests
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_community.document_loaders import TextLoader
#from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.storage import LocalFileStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.tools import ToolException
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain
from langchain.chains import LLMChain
from langchain.chains import SequentialChain, TransformChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import argparse

parser = argparse.ArgumentParser(description='AI LLM parser.')
parser.add_argument('--question', type=str, help='Ask a question or send a command to LLM')
args = parser.parse_args()
#print(args.question)
HUBITAT_TOKEN = os.getenv("HUBITAT_TOKEN")

# Decide whether to home automation chain or agent
chain = (
    PromptTemplate.from_template(
        """Given the user question below, classify it as either being about `home automation`, or `Other`.

Do NOT respond with anything other than the 2 options specified.  Do NOT respond with how the Classification was selected.

<question>
{question}
</question>

Classification:"""
    )

    | ChatOllama(model="mistral")
    | StrOutputParser()
)

chain_result = chain.invoke({"question": f"{args.question}"})
#print(chain_result)

DEVICE_LOOKUP_TEMPLATE_TEXT = """
Parse the device JSON data to determine which device the user is asking about in the Question.
Device names will not always be exact, select the closest match(es).
Devices:
{context}

Respond only with the Device ID or IDs in a Python list form.  Ex. [12, 4] or Ex. [6]

Question: {input}
Answer:
"""

#Ex. [{{'id': 7, 'command': 'off'}, {'id': 17, 'command': 'Heat'}]
DETERMINE_COMMAND_TEMPLATE_TEXT = """
Determine the action that should be taken OR the status response based on the Question.
Respond in 1 of 3 ways:
1. Command action: Parse the JSON data, select a command or commands from the command sections that satisfy the action(s) from the Question.
Respond ONLY with a list of JSON object(s) formatted with the Device ID(s) and the command(s).
The command may require an additional argument.
Ex. [{{'id': 7, 'command': 'A_SUPPORTED_COMMAND', 'value': 'A_VALUE}}]
Ex. [{{'id': 7, 'command': 'A_SUPPORTED_COMMAND'}}]
Ex. [{{'id': 7, 'command': 'A_SUPPORTED_COMMAND'}}, {{'id': 17, 'command': 'A_SUPPORTED_COMMAND'}}]

2. Status asking about a specific attribute: Parse the JSON data, if the Question does not contain an action that would change the state of a device, respond only with the value or values of the attribute(s) from the device data.
Do NOT mention the JSON data or the Question in the response.
Ex. The TV appears to be on.
Ex. The temperature of the downstairs thermostat is 75 degrees.

3. Status without asking for a specific attribute: Parse the JSON data.  If the Question does not ask about a specific attribute summarize the currentValue of all of them.
Respond only with the device and device information, do NOT mention the JSON data or the Question in the response.
Ex. The Dining Room Lights are on and the level is 90.

JSON data:
{output_text} 

Question: {input}
Answer:
"""

#llm = Ollama(model="mistral:7b-instruct-q4_0")
llm = Ollama(model="mistral", temperature=0.1)
#llm = Ollama(model="llama3")
params = {
    'access_token': f'{HUBITAT_TOKEN}'
}
response = requests.request("get", "http://192.168.2.9/apps/api/106/devices", params=params)

def transform_func(inputs: dict) -> dict:
    """
    Take the specified dictionary which contains a device or list of devices and get more specific details
    """
    print(f"the inputs: {inputs['updatedcontext']}")
    clean_inputs = eval(inputs['updatedcontext'].strip(" "))
    response_list = []
    for i in clean_inputs:
        print(i)
        params = {'access_token': f'{HUBITAT_TOKEN}'}
        response = requests.request("get", f"http://192.168.2.9/apps/api/106/devices/{i}", params=params)

    return {"output_text": response.text}

def transform_output(inputs: dict) -> dict:
    """
    Take the specified dictionary and execute a command or return device details
    """
    " [{'id': 116, 'command': 'off'}]"
    #print(f"the inputs: {inputs['answer']}")
    try:
        clean_inputs = eval(inputs['answer'].lstrip(" "))
    except:
        clean_inputs = inputs['answer'].lstrip(" ")
    if type(clean_inputs) == list:
        for i in clean_inputs:
            device = i['id']
            command = i['command']
            if 'value' in i: 
                value = i['value']
                print(f"Running {command} and {value} on {device}")
                params = {'access_token': f'{HUBITAT_TOKEN}'}
                response = requests.request("get", f"http://192.168.2.9/apps/api/106/devices/{device}/{command}/{value}", params=params)
            else:
                print(f"Running {command} on {device}")
                params = {'access_token': f'{HUBITAT_TOKEN}'}
                response = requests.request("get", f"http://192.168.2.9/apps/api/106/devices/{device}/{command}", params=params)
    else:
        print(clean_inputs)
    #print(f"all inputs: {inputs}")
    return {"answernew": "something"}

# Create the individual prompt templates.
#DEVICE_LOOKUP_template = PromptTemplate.from_template(DEVICE_LOOKUP_TEMPLATE_TEXT)
DEVICE_LOOKUP_template = PromptTemplate(input_variables=["input", "context"], template=DEVICE_LOOKUP_TEMPLATE_TEXT)
DETERMINE_COMMAND_template = PromptTemplate(input_variables=["input", "output_text"], template=DETERMINE_COMMAND_TEMPLATE_TEXT)

# Create the chains - Device action and Device status
device_lookup_chain = LLMChain(llm=llm, prompt=DEVICE_LOOKUP_template, output_key="updatedcontext", verbose=True, output_parser=StrOutputParser())
determine_command_chain = LLMChain(llm=llm, prompt=DETERMINE_COMMAND_template, output_key="answer", verbose=True, output_parser=StrOutputParser())
transform_chain = TransformChain(
    input_variables=["updatedcontext"], output_variables=["output_text"], transform=transform_func, verbose=True
)
output_chain = TransformChain(
    input_variables=["answer"], output_variables=["answernew"], transform=transform_output, verbose=True
)


# # Create the chains - Historical
# device_lookup_chain = LLMChain(llm=llm, prompt=DEVICE_LOOKUP_template, output_key="updatedcontext", verbose=True, output_parser=StrOutputParser())
# determine_command_chain = LLMChain(llm=llm, prompt=DETERMINE_COMMAND_template, output_key="answer", verbose=True, output_parser=StrOutputParser())
# transform_chain = TransformChain(
#     input_variables=["updatedcontext"], output_variables=["output_text"], transform=transform_func, verbose=True
# )
# output_chain = TransformChain(
#     input_variables=["answer"], output_variables=["answernew"], transform=transform_output, verbose=True
# )
influx_token = os.getenv("INFLUX_TOKEN")
headers = {'Accept': 'applicaton/csv', 'Content-type': 'application/vnd.flux', 'Authorization': f'Token {influx_token}'}
data = '''from(bucket: "Hubitat")
    |> range(start: -60m)
    |> filter(fn: (r) => r["_measurement"] == "contact")
    |> filter(fn: (r) => r["_field"] == "valueBinary")'''
influxdb_data = requests.request("POST", "http://192.168.2.184:8086/api/v2/query?org=Hubitat", headers=headers, data=data)



# Join them into a sequential chain.
overall_chain = SequentialChain(
    chains=[device_lookup_chain, transform_chain, determine_command_chain, output_chain],
    input_variables=["input", "context"],
    # Here we return multiple variables
    #output_variables=["updatedcontext", "answer"],
    #output_variables=["answernew"],
    verbose=True,
    return_all=True)

all_devices = response.json()

if chain_result == " home automation.":
    overall_chain_result = overall_chain.invoke({"input":f"{args.question}", "context": all_devices})
    #overall_chain.invoke({"input":"if the kitchen lights are off, turn them on", "context": all_devices})
    #overall_chain.invoke({"input":"are the kitchen lights on or off?", "context": all_devices})
    #overall_chain.invoke({"input":"what is the status of the upstairs thermostat?", "context": all_devices})
    #overall_chain.invoke({"input":"what mode is the downstairs thermostat set to?", "context": all_devices})
    #overall_chain.invoke({"input":"set the downstairs hallway lights to 100%", "context": all_devices})
    print(overall_chain_result)
else:
    # Decide whether to home automation chain or agent
    chain = (
        PromptTemplate.from_template(
            """You are a helpful assistant, answer the question to the best of your ability.
        <question>
        {question}
        </question>

        Answer:"""
        )
        | ChatOllama(model="mistral")
        | StrOutputParser()
    )

    chain_result = chain.invoke({"question": f"{args.question}"})
    print(chain_result)



