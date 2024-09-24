from langchain_community.document_loaders import JSONLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
import argparse
import os
import requests
import json

parser = argparse.ArgumentParser(description="AI app6")

# Add arguments
parser.add_argument('--question', type=str, required=True)

HUBITAT_TOKEN = os.getenv("HUBITAT_TOKEN")

params = {
    'access_token': f'{HUBITAT_TOKEN}'
}

response = requests.request("get", "http://192.168.2.9/apps/api/106/devices", params=params)
all_devices = response.json()
f = open("./data/devices.json", "w")
f.write(json.dumps(all_devices))
f.close()

for i in all_devices:
    response = requests.request("get", f"http://192.168.2.9/apps/api/106/devices/{i['id']}", params=params)
    device = response.json()
    f = open(f"./data/device{i['id']}.json", "w")
    f.write(json.dumps(device))
    f.close()

# Parse the arguments
args = parser.parse_args()

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#embedding_function = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")

# get our list of json objects - fix Nulls
file_path = "./data/devices.json"
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()
    content = content.replace('null','""')
    #print(content)

raw_documents = eval(content)

# get our list of json objects - fix Nulls
#file_path = "./data/device116.json"
# file_path = "./data/device340.json"
# with open(file_path, 'r', encoding='utf-8') as file:
#         content = file.read()

path = "./data"
file_type = ".json"
files = []
for file in os.listdir(path):
    if file.endswith(file_type):
        files.append(file)
file_content = []
this = []
for i in files:
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        content = content.replace('null','""')
        raw_documents2 = eval(content)
        this.append(raw_documents2)


# set text and meta data for each json object in the list
preprocessed_documents = [
    {
        "text": f"Device ID: {doc['id']}, Name: {doc['name']}, Label: {doc['label']}, Type: {doc['type']}, Room: {doc['room']}",
        "metadata": doc
    } for doc in raw_documents
]


import json

def preprocess_documents_for_chroma(documents):
    processed_documents = []

    for doc in documents:
        # Initialize attributes as an empty dictionary
        attributes = {}
        
        # Check if 'attributes' is a list of dictionaries
        if isinstance(doc.get('attributes'), list):
            attributes = {attr['name']: attr['currentValue'] for attr in doc['attributes'] if isinstance(attr, dict)}
        
        # Flatten capabilities, handle nested attributes
        # capabilities = []
        # if isinstance(doc.get('capabilities'), list):
        #     for cap in doc['capabilities']:
        #         if isinstance(cap, dict) and 'attributes' in cap :
        #             capabilities.extend([attr['name'] for attr in cap['attributes'] if isinstance(attr, dict)])
        #         elif isinstance(cap, str):
        #             capabilities.append(cap)
        capabilities = []
        if isinstance(doc.get('capabilities'), list):
            for cap in doc['capabilities']:
                if isinstance(cap, dict) and 'attributes' in cap:
                    for attr in cap['attributes']:
                        if isinstance(attr, dict) and attr['name'] not in ['ledOn', 'ledOff']:
                            capabilities.append(attr['name'])
                elif isinstance(cap, str) and cap not in ['ledOn', 'ledOff']:
                    capabilities.append(cap)
        
        # Create a processed document with flattened and serialized fields
        processed_doc = {
            "id": doc.get("id"),
            "name": doc.get("name"),
            "label": doc.get("label"),
            "type": doc.get("type"),
            "room": doc.get("room"),
            "attributes": json.dumps(attributes),  # Convert dict to JSON string
            "capabilities": json.dumps(capabilities),  # Convert list to JSON string
            "commands": json.dumps(doc.get("commands", []))  # Convert list to JSON string
        }
        
        processed_documents.append(processed_doc)
    
    return processed_documents

def format_for_chroma(processed_documents):
    formatted_documents = []

    for doc in processed_documents:
        # Create the text representation
        text_representation = f"Device {doc.get('name')} ({doc.get('label')}) in {doc.get('room')} is a {doc.get('type')}."
        
        # Format the metadata as a JSON string
        metadata_representation = json.dumps(doc)
        
        # Create the final document structure
        formatted_doc = {
            "text": text_representation,
            "metadata": metadata_representation
        }
        
        formatted_documents.append(formatted_doc)
    
    return formatted_documents

#print(raw_documents2)
#print(f"{type(raw_documents2)}")

processed_data = preprocess_documents_for_chroma(this)
#print(processed_data)

processed_data2 = format_for_chroma(processed_data)
#print(processed_data2)
this2 = []
this2.append(processed_data2)
documents2 = [
    Document(page_content=doc["text"], metadata=json.loads(doc["metadata"]))
    for doc in processed_data2
]

preprocessed_documents = [
    {
        "text": f"Device ID: {doc['id']}, Name: {doc['name']}, Label: {doc['label']}, Type: {doc['type']}, Room: {doc['room']}",
        "metadata": doc
    } for doc in raw_documents
]
#print(documents2)

documents = [
    Document(page_content=doc["text"], metadata=doc["metadata"])
    for doc in preprocessed_documents
]

merged_docs = []
merged_docs = documents + documents2

db = Chroma.from_documents(merged_docs, embedding_function)

# # Perform a similarity search
#query = "What commands do the Kitchen Can Lights support?"
query = args.question
results = db.similarity_search(query)

# # Print the results
# for result in results:
#     print(result)

### add the model ###

retriever = db.as_retriever(search_type="similarity")

## chain based ##
# template = """Answer the question based only on the following context:
# {context}

# Question: {question}
# """
# prompt = ChatPromptTemplate.from_template(template)

model = Ollama(model="mistral", temperature=0.1, base_url="http://192.168.2.218:11434")
#model = Ollama(model="llama3.1", base_url="http://192.168.2.218:11434")
llm = model

# chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | prompt
#     | model
#     | StrOutputParser()
# )

# query = args.question
# print(chain.invoke(query))

from langchain.tools.retriever import create_retriever_tool
import requests
import csv
from langchain.agents import load_tools
from langchain_community.llms import Ollama
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, create_tool_calling_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import ToolException
from langchain_core.prompts import PromptTemplate
from langchain.tools import BaseTool, StructuredTool, tool

retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="IoT Device Information",
    description="Gets the documents which contain an IoT device list and commands associated with those devices",
)

def _handle_error(error: ToolException) -> str:
    return (
        "The following errors occurred during tool execution:"
        + error.args[0]
        + "Please retry this tool."
    )

class DeviceInput(BaseModel):
    jsonargs: str = Field(description="""Pass a valid command to a device - {{"id": "value", "command": "value"}}""",default="cmd")
#     #deviceid: str = Field(description="should be the identifier of a device from another tool",default="0")

# class DeviceInput(BaseModel):
#     jsonargs: str

def substring_match_command(input_command, valid_commands):
    for valid_command in valid_commands:
        if valid_command in input_command:
            return valid_command
    return None

def SendCommandToIOTDevice(jsonargs: str) -> dict:
    """Send a command to a specific IoT device by ID. kwargs should match this format {{"id": "value", "command": "value"}}"""
    #query = int(query)
    #print(deviceid)
    #print(command)
    #jsonargs = f"{'id': {id}, 'command': {command}}"
    print(f"These are the arguments sent to the tool: {jsonargs}")
    jsonargs = eval(jsonargs)
    # if command['deviceid'] == 0:
    #     return "Please enter a valid DeviceID"
    # if command['action'] == "cmd":
    #     return "Please enter a valid Device Command"
    params = {
    'access_token': '07f4a1a4-ab47-428e-9ef9-0df31e251e58'
    }
    if 'id' or 'device_id' or 'device\_id' in jsonargs.keys():
        print("yes")
        print(dir(jsonargs.keys()))
        jsonargs['id'] = jsonargs[f"{eval(str(jsonargs.keys()).lstrip('dict_keys(').rstrip(')'))[0]}"]
    else:
        raise ToolException('Error: Provided arguments are not correct, please use format: {{"id": "value", "command": "value"}}')
    if 'command' in jsonargs.keys():
        print("yes")
    else:
        raise ToolException('Error: Provided arguments are not correct, please use format: {{"id": "value", "command": "value"}}')
    
    deviceid = int(jsonargs['id'])
    command = jsonargs['command']
    response = requests.get(f"http://192.168.2.9/apps/api/106/devices/{deviceid}", params=params)
    available_commands = response.json()['commands']
    print(available_commands)
    match_command = substring_match_command(command, available_commands)
    if match_command:
        print("Command matched:", match_command)
        command = match_command
    else:
        print("Command not valid.")
    if command in available_commands:
        print("Specified command is valid")
    else:
        raise ToolException(f"Error: The specified command: {command} is not valid for this device")
    
    response = requests.get(f"http://192.168.2.9/apps/api/106/devices/{deviceid}/{command}", params=params)
    response = {"action": "sucess"}
    return response

SendCommand = StructuredTool.from_function(
    func=SendCommandToIOTDevice,
    name="Send Command To IOT Device",
    description="send a supported command to an IOT device",
    args_schema=DeviceInput,
    return_direct=True,
    handle_tool_error=_handle_error
    # coroutine= ... <- you can specify an async method if desired as well
)

# GetAvailableCommands = StructuredTool.from_function(
#     func=GetIOTDeviceCommand,
#     name="Send Command To IOT Device",
#     description="send a supported command to an IOT device",
#     args_schema=DeviceInput,
#     return_direct=True,
#     handle_tool_error=_handle_error
#     # coroutine= ... <- you can specify an async method if desired as well
# )
prompt = hub.pull("hwchase17/react")

template_mod = """
You are a home assistant agent which interacts with devices using provided tools
ONLY use provided tools for IoT related actions.
Refer to Class and tool descriptions for proper tool input formats.

You have access to the following tools:

{tools}

Use the following format:

Question: the action you should complete
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"id": "value", "command": "value"}})

Begin!

Question: {input}
Thought:{agent_scratchpad}

"""

# template_mod =  """
# You are an intelligent home assistant specialized in IoT device management. Your primary responsibility is to interact with devices by sending precise commands based on provided documents. 

# ### Key Rules:
# - **No Code or Scripts**: Do not provide code or scripts in your responses.
# - **Tool Usage Only**: You are limited to using the provided tools for all IoT-related actions.
# - **Strict Command Matching**: Always refer to the device command metadata. Send commands exactly as specified in the documents.
# - **Contextual Awareness**: Use Labels and Rooms in the metadata to correctly identify devices and their locations, all commands should be completed using the id of the device.

# ## Available Tools:
# {tools}

# ### Response Structure:
# - **Question**: The input query or action to be performed.
# - **Thought**: Reflect on the best approach to solve the problem.
# - **Action**: Choose an appropriate action from [{tool_names}].
# - **Action Input**: Provide the input for the tool in valid JSON format (e.g., {{"id": "value", "command": "value"}}).
# - **Observation**: Record the outcome of the action.
# - **Final Answer**: Conclude with a direct answer to the original question.

# Begin!

# Question: {input}
# Thought: {agent_scratchpad}
# """

# template_mod =  """
# You are an intelligent home assistant specialized in IoT device management. Your primary responsibility is to interact with devices by sending precise commands based on provided information.

# ### Instructions:
# 1. **Actions First**: If an action is required, only provide the action and action input. Do not include a final answer until all actions are completed.
# 2. **Final Answer Only After Actions**: Once all actions have been successfully executed, you will receive a prompt or flag to provide the final answer.
# 3. **No Overlapping**: Do not mix actions and the final answer in the same response. Provide actions first, then wait for a prompt before giving the final answer.

# ### Response Format:
# 1. Action: [Action Name] Choose an appropriate action using a tool [{tool_names}]
# 2. Action Input: [JSON formatted input]
# 3. Observation: [Result of the action]
# ... (Repeat the above steps as necessary)
# 4. Final Answer: [The final conclusion after all actions are completed]

# ## Available Tools:
# {tools}

# Begin!

# Question: {input}
# Thought: {agent_scratchpad}

# """

#tools = [retriever_tool, SendCommand]
tools = [SendCommand]

prompt = PromptTemplate.from_template(template_mod)
agent = create_react_agent(llm, tools, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
#turn_kitchen_light_on = "http://192.168.2.9/apps/api/106/devices/116/on?access_token=07f4a1a4-ab47-428e-9ef9-0df31e251e58"
#agent_executor.invoke({"input": "hi, use the requests_get tool to get information from wikipedia about abraham lincoln"})
#agent_executor.invoke({"input": "what is the device ID for the Downstairs Hallway Lights?"})
#agent_executor.invoke({"input": f"{args.question}"})


from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(f"Metadata: {doc.metadata}\nContent: {doc.page_content}" for doc in docs)

prompt = hub.pull("rlm/rag-prompt")
prompt.messages[0].prompt.template = """
You are an assistant for question-answering tasks regarding IoT devices. 
Use ONLY the retrieved context, particularly focusing on the metadata, to answer the question. 
Keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# retrieved_docs = retriever.invoke(f"{args.question}")
# #print(retrieved_docs)
# # for chunk in rag_chain.stream(f"{args.question}"):
# #     print(chunk, end="", flush=True)
# print(rag_chain.invoke(f"{args.question}"))


class Classification(BaseModel):
    prompt_tag: str = Field(description="Is the text asking for information or asking for an action to be taken?")
    


tagging_prompt = ChatPromptTemplate.from_template(
    """
    You will be provided with a human input. Determine whether the input is asking for information or requesting an action to be taken.
    Respond in the following JSON format:
    {{"prompt_tag": "<information|action>"}}
    
    Human input:
    {input}
    """
)

def classify_input(input_text):
    # Instantiate the Ollama model without structured output
    llm = Ollama(temperature=0, model="mistral", base_url="http://192.168.2.218:11434")
    
    # Create the chain with the tagging prompt and LLM
    chain = tagging_prompt | llm
    
    # Invoke the chain with the provided input text
    result = chain.invoke({"input": input_text})
    
    # Parse the result to extract the classification
    try:
        classification_result = json.loads(result)['prompt_tag']
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError("Failed to parse the classification result.") from e
    
    return classification_result

# Step 2: Define the chain that handles questions
def handle_question_chain():
    chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    result = chain.invoke(f"{args.question}")
    return result

# Step 3: Define the agent executor for actions
def handle_action_chain():
    chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    chain_result = chain.invoke(f"{args.question}")
    print(chain_result)
    try:
        result = agent_executor.invoke({"input": f"{chain_result}"})
    except Exception as e:
        #loop until success
        print(f"Got an error, restarting the chain {e}")
        handle_action_chain()
    return result

# Step 4: Define the logic chain execution (LCEL) chain
def lcel_chain(input_text):
    # Step 4a: Classify the input
    input_type = classify_input(input_text)
    
    # Step 4b: Execute the appropriate chain based on the classification
    if input_type == '<information>':
        print("Selected information chain")
        return handle_question_chain()
    elif input_type == '<action>':
        print("Selected action chain")
        return handle_action_chain()
    else:
        return "Unable to classify input. Please try again."

# Example usage:

response = lcel_chain(args.question)
print(response)