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

HUBITAT_TOKEN = os.getenv("HUBITAT_TOKEN")

# chain = (
#     PromptTemplate.from_template(
#         """Given the user question below, classify it as either being about `home automation`, or `Other`.

# Do not respond with anything other than the 2 options specified.

# <question>
# {question}
# </question>

# Classification:"""
#     )

#     | ChatOllama(model="mistral")
#     | StrOutputParser()
# )

#chain.invoke({"question": "how do I call Anthropic?"})

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
    Extracts specific sections from a given text based on newline separators.

    The function assumes the input text is divided into sections or paragraphs separated
    by one newline characters (`\n`). It extracts the sections from index 922 to 950
    (inclusive) and returns them in a dictionary.

    Parameters:
    - inputs (dict): A dictionary containing the key "text" with the input text as its value.

    Returns:
    - dict: A dictionary containing the key "output_text" with the extracted sections as its value.
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
    Extracts specific sections from a given text based on newline separators.

    The function assumes the input text is divided into sections or paragraphs separated
    by one newline characters (`\n`). It extracts the sections from index 922 to 950
    (inclusive) and returns them in a dictionary.

    Parameters:
    - inputs (dict): A dictionary containing the key "text" with the input text as its value.

    Returns:
    - dict: A dictionary containing the key "output_text" with the extracted sections as its value.
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

# Create the chains.
device_lookup_chain = LLMChain(llm=llm, prompt=DEVICE_LOOKUP_template, output_key="updatedcontext", verbose=True, output_parser=StrOutputParser())
determine_command_chain = LLMChain(llm=llm, prompt=DETERMINE_COMMAND_template, output_key="answer", verbose=True, output_parser=StrOutputParser())
transform_chain = TransformChain(
    input_variables=["updatedcontext"], output_variables=["output_text"], transform=transform_func, verbose=True
)
output_chain = TransformChain(
    input_variables=["answer"], output_variables=["answernew"], transform=transform_output, verbose=True
)

#transform_chain.run(meditations)


# Join them into a sequential chain.
# overall_chain = SimpleSequentialChain(
#     chains=[device_lookup_chain, determine_command_chain], verbose=True
# )
overall_chain = SequentialChain(
    chains=[device_lookup_chain, transform_chain, determine_command_chain, output_chain],
    input_variables=["input", "context"],
    # Here we return multiple variables
    #output_variables=["updatedcontext", "answer"],
    #output_variables=["answernew"],
    verbose=True,
    return_all=True)

all_devices = response.json()

overall_chain.invoke({"input":"if the kitchen lights are on, turn them off", "context": all_devices})
#overall_chain.invoke({"input":"if the kitchen lights are off, turn them on", "context": all_devices})
#overall_chain.invoke({"input":"are the kitchen lights on or off?", "context": all_devices})
#overall_chain.invoke({"input":"what is the status of the upstairs thermostat?", "context": all_devices})
#overall_chain.invoke({"input":"what mode is the downstairs thermostat set to?", "context": all_devices})
#overall_chain.invoke({"input":"set the downstairs hallway lights to 100%", "context": all_devices})
