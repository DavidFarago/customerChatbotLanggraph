"""# Customer Chatbot Flow

Respond to new customer message:
1. messages = [new customer message] # list of Customer message or MediVoice message
2. expectation detection based on messages:
  * "response needed"
  * "no response needed".

If "no response needed", go to 10, else go to 3.
3. use last customer message (and last queries/keywords) to create queries/keywords for query rewriting RAG
4. retrieve chunks based on last customer message and last messages and created queries or keywords
5. missing info detection based on the data:
  * "needs further customer info",
  * "sufficient customer info".
6. use messages, retrieved chunks and missing info detection output = (and possibly helpfulness critique) to write a draft reply
7. check draft reply for helpfulness: is the draft reply helpfull in moving the problem forward (e.g. by answering the customer inquery or by asking the customer for further information that will help move the problem forward)?
  * No: store helpfulness critique and goto 8.
  * Yes: go to 9.
8. Manual intervention detection based on the data:
  * "human needed"
  * "no human needed"

If human needed or loop counter > RETRIES: go to 11.
Else increase loop counter and go to 6.
9. Respond to the customer with the generated message and wait for new customer message. Upon new customer message or timeout, go to 2.
10. issue tracker item detection:
  * "issue tracker item needed": there should be an issue tracker item, but is not yet
  * "no issue tracker item needed": it is not worth to create an issue tracker item for these messages, or an issue tracker tem is already present in the messages.

If issue tracker item needed, goto 11, else goto 12.
11. Create an issue tracker item, send the customer a reply with the item ID, and go to 12.
12. If there is an issue item: {store all information in the issue item}. Terminate.

"""
#%%
from enum import Enum
import os
from typing import Annotated, List
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate

TOP_K_FUSION_QUERIES = 3
TOP_K_CHUNKS = 3
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIES = 3

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

initial_customer_messages = [
    "Why is my phone assistant not offering any appointments in Febuary?",
    "How do I configure multiple locations for my practice?",
    "Can I configure my assistant to understand Spanish besides German?",
    "Your system is great, keep up the good work!",
    "We stopped using your assistant and canceled our contract with MediVoice. Goodbye.",
    "I have a bug report: Your assistant does not schedule appointments on Sundays, even though I configured slots for Sundays.",
    "I get back different names for the same person, in the same call :-0",
]

customer_service_sys_msg_prefix = """
    You are the support agent for a customer service chatbot for MediVoice, a telefone bot offered to medical practices. Therefore
    * MediVoice is offering voice assistants that can autonomously manage patient inqueries about their appointments, prescriptions, transfer slips and the like
    * MediVoice customers are medical practicess, medical care centers, phone centers, and the like, who are offering a telefone bot for their patients.
"""
classification_sys_msg_prefix = """You are an expert customer message classification agent. You are perfect at
    * understanding what a customer wants when they write a message
    * categorizing customer messages into the given classes.
"""

"""# Basic Chains

1. Expectation detection: expectation_classifier_chain
2. Create queries or keywords: rag_fusion_chain
3. Retrieve chunks: rag_chain
4. Missing info detection: missing_info_classifier_chain
5. Write a draft reply: response_draft_chain
6. Helpfulness detection: helpfulness_classifier_chain
7. Manual intervention detection: intervention_classifier_chain
8. Issue tracker item detection: whether there is resp. should be an issue tracker item.
"""

#%%
print("# Expectation detection")

class ExpectationClasses(str, Enum):
    NEEDS = "response needed"
    NO_RESPONSE = "no response needed"
class ExpectationClassification(BaseModel):
    explanation: str = Field(description="An explanation for the classification result")
    category: ExpectationClasses = Field(description="The classification result")
structured_llm = llm.with_structured_output(ExpectationClassification, method="json_schema")
expectation_detection_sys_msg = ("system", customer_service_sys_msg_prefix + classification_sys_msg_prefix + """
    Conduct a comprehensive step by step analysis of the list of messages provided below. Then categorize the list of messages into one of the following given classes:
    * "response needed": the customer expects a response for the list of messages because he asked a question that is not yet answered, he reported a bug that is not yet fixed, or he wrote about a problem where a new response can fix or reduce his problem
    * "no response needed": the customer does NOT expect a response because he just gave some feedback, was just venting without expecting response, wrote something off-topic, or got already all responses he needed to resolve his problem.
    Output nothing but one of the classes, i.e. "response needed" or "no response needed", nothing else.
    """)
expectation_detection_prompt = ChatPromptTemplate.from_messages([expectation_detection_sys_msg, ("user", "{messages}")])
expectation_classifier_chain = expectation_detection_prompt | structured_llm

def get_messages_str(messages: List[str]) -> str:
    return '\n'.join(messages)

# exemplary use of expectation_classifier_chain:
messages = [f"Customer message: {initial_customer_messages[0]}"] # part of graph state
print("Starting off with customer message: {initial_customer_messages[0]}")
print(expectation_classifier_chain.invoke({"messages": get_messages_str(messages)}))

#%%
print("# Create queries or keywords")

class AlternativeQueries(BaseModel):
    explanation: str = Field(description="An explanation for all the alternative queries/keywords")
    queries: Annotated[List[str], TOP_K_FUSION_QUERIES] = Field(description="The alternative queries/keywords")
structured_llm = llm.with_structured_output(AlternativeQueries, method="json_schema")

create_queries_keywords_sys_msg = ("system", customer_service_sys_msg_prefix + """
    You are a RAG fusion expert, i.e. a master at creating alternative queries/keywords that are related to the information you will be provided: a customer message and possibly earlier alternative queries/keywords.
    Create alternative queries/keywords that rephrase the provided customer message to perfect the retrieval of chunks relevant to the provided customer message from a vector database.

    First think step by step to work out the best alternative queries/keywords that are closely related to the customer message (and possibly earlier alternative queries/keywords) provided below, to increase the relevance of the retrieved chunks for the provided customer message.
    Each alternative can be a query or a keyword, whatever you think is better. The provided customer message and each of your generated alternative queries/keywords will be used separately to retrieve relevant chunks from a vector database.

    Generate a JSON with the keys
    * `explanation` for your step by step thought on what the best 3 alternative queries or keywords are
    * `queries` for a list of {number_queries} alternative queries/keywords.""")
alternative_queries_prompt = ChatPromptTemplate.from_messages([create_queries_keywords_sys_msg, ("user", "{last_customer_message}\n{previous_queries}")])
alternative_queries_chain = alternative_queries_prompt | structured_llm

def get_last_customer_message_str(messages: List[str]) -> str:
  return next(item for item in reversed(messages) if item.startswith("Customer"))

def get_optional_previous_queries_str(previous_queries: AlternativeQueries) -> str:
  if not previous_queries:
    return ''
  return '\n'.join([f'Previous query/keyword {i}: {query}' for i, query in enumerate(previous_queries.queries)])

# exemplary use of alternative_queries_chain:
previous_queries: AlternativeQueries = None
alternative_queries = alternative_queries_chain.invoke({
    "number_queries": TOP_K_FUSION_QUERIES,
    "last_customer_message": get_last_customer_message_str(messages),
    "previous_queries": get_optional_previous_queries_str(previous_queries),
}) # part of graph state
print(alternative_queries)

#%%
print("# Retrieve chunks")

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

ChunkArray = Annotated[List[str], TOP_K_CHUNKS * (TOP_K_FUSION_QUERIES + 1)]

markdown_path = "docs"
markdown_files = [os.path.join(markdown_path, f) for f in os.listdir(markdown_path) if f.endswith(".md")]

data = []
for file_path in markdown_files:
    loader = UnstructuredMarkdownLoader(file_path)
    data.extend(loader.load())
print(f"Loaded {len(data)} documents")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
texts = text_splitter.split_documents(data)
openai_embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)

print(f"Splittet into {len(texts)} chunks")

from langchain_chroma import Chroma
persist_directory = 'db'
vectordb = Chroma.from_documents(documents=texts,
                                 embedding=openai_embeddings,
                                 persist_directory=persist_directory)
retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K_CHUNKS})

# exemplary use of chunk:
chunks_collector = []
for query in [get_last_customer_message_str(messages)] + alternative_queries.queries:
    chunks_collector.extend(retriever.invoke(query))
chunks: ChunkArray = {chunk.page_content for chunk in chunks_collector} # part of graph state
print(f"Retrieved {len(chunks)} chunks: {chunks}")

#%%
print("# Missing info detection")

class CustomerInfoClasses(str, Enum):
    MORE_INFO = "needs more customer info"
    ENOUGH_INFO = "enough customer info"
class CustomerInfoClassification(BaseModel):
    explanation: str = Field(description="An explanation for the classification result")
    category: CustomerInfoClasses = Field(description="The classification result")
structured_llm = llm.with_structured_output(CustomerInfoClassification, method="json_schema")

missing_customer_info_detection_sys_msg = ("system", customer_service_sys_msg_prefix + classification_sys_msg_prefix + """
   Conduct a comprehensive step by step analysis for the classification, using the data provided below: a list of messages plus chunks from MediVoice's vector database that are related to the last customer message.
   Then classify the provided data into one of the following given classes:
    * "needs more customer info": more information needs to be asked from the customer to be able to give him a helpful response, i.e. to fix or reduce his problem
    * "enough customer info": no more information is needed from the customer.
    Output nothing but one of the classes, i.e. "needs more customer info" or "enough customer info", nothing else.
    """)
missing_info_detection_prompt = ChatPromptTemplate.from_messages([missing_customer_info_detection_sys_msg, ("user", "{messages}\n{chunks}")])
missing_info_classifier_chain = missing_info_detection_prompt | structured_llm

def get_chunks_str(chunks: ChunkArray) -> str:
    return '\n'.join([f'Chunk {i}: {chunk}' for i, chunk in enumerate(chunks)])

# exemplary use of missing_info_classifier_chain:
missing_customer_info = missing_info_classifier_chain.invoke({
    "messages": get_messages_str(messages),
    "chunks": get_chunks_str(chunks)
}) # part of graph state
print(missing_customer_info)

#%%
print("# Write response draft")

class ResponseDraft(BaseModel):
    explanation: str = Field(description="An explanation for the response draft")
    responseDraft: str = Field(description="The response draft")
structured_llm = llm.with_structured_output(ResponseDraft, method="json_schema")

response_draft_sys_msg = ("system", customer_service_sys_msg_prefix + """
    You are an expert response writer for customer requests.
    Conduct a step by step analysis on how to respond, using the data provided below:
    * a list of messages, showing the conversation between MediVoice and the customer so far
    * chunks from MediVoice's vector database that are related to the last customer message, showing the most relevant parts of MediVoice's documentation
    * missing information detection result, showing whether the customer response should aks the customer for missing information
    * optionally an earlier draft for a reply to the customer, together with a reasoning why the draft is not helpful for solving the customer's problem.
    Then write the response to the customer that moves the conversation forward to solving the customer's problem.
    But never make up functionality of MediVoice that is not contained in the data provided to you.""")
response_draft_prompt = ChatPromptTemplate.from_messages([response_draft_sys_msg, ("user", "{messages}\n{chunks}\n{missing_info}\n{response_draft}\n{unhelpfulness_reason}")])
response_draft_chain = response_draft_prompt | structured_llm

def get_missing_info_str(customer_info: CustomerInfoClassification) -> str:
    return f'\nMissingCustomerInfo verdict: {repr(customer_info.category)}\nMisingCustomerInfo explanation: {customer_info.explanation}'

# exemplary use of response_draft_chain:
response_draft = response_draft_chain.invoke({
    "messages": get_messages_str(messages),
    "chunks": get_chunks_str(chunks),
    "missing_info": get_missing_info_str(missing_customer_info),
    "response_draft": "",
    "unhelpfulness_reason": "",
}) # part of graph state
print(response_draft)

#%%
print("# Helpfulness detection")

class HelpfulnessClasses(str, Enum):
    HELPFUL = "response draft is helpful for the customer"
    NOT_HELPFUL = "response draft is not helpful for the customer"
class HelpfulnessClassification(BaseModel):
    explanation: str = Field(description="An explanation for the classification result")
    category: HelpfulnessClasses = Field(description="The classification result")
structured_llm = llm.with_structured_output(HelpfulnessClassification, method="json_schema")

helpfulness_detection_sys_msg = ("system", customer_service_sys_msg_prefix + classification_sys_msg_prefix + """
   Conduct a comprehensive step by step analysis for the classification, using the data provided below:
   * a list of messages, showing the conversation between MediVoice and the customer so far
   * chunks from MediVoice's vector database that are related to the last customer message
   * missing information detection result, showing whether the customer response should aks the customer for missing information
   * a draft for a reply to the customer.
   The step by step analysis should inspect the draft reply by first summarizing it, then looking at all the other provided data, and then give a judgement whether the draft reply is helpful, 
   i.e. whether it gives new information to the customer, or asks for new information from the customer.
   Then do a classification into one of the following given classes:
    * "response draft is helpful for the customer": if it gives new information to the customer, or asks for new information from the customer.
    * "response draft is not helpful for the customer": otherwiese.
    For the classification, output nothing but one of the classes, i.e. "response draft is helpful for the customer" or "response draft is helpful for the customer", nothing else.
    """)
helpfulness_detection_prompt = ChatPromptTemplate.from_messages([helpfulness_detection_sys_msg, ("user", "{messages}\n{chunks}\n{missing_info}\n{response_draft}")])
helpfulness_classifier_chain = helpfulness_detection_prompt | structured_llm

def get_earlier_response_draft_str(response_draft: ResponseDraft) -> str:
    return f'\nEarlier response draft: {response_draft.responseDraft}'

def get_unhelpful_reason_str(helpfulness_classification: HelpfulnessClassification) -> str:
    assert helpfulness_classification.category == HelpfulnessClasses.NOT_HELPFUL
    return f'\nReasoning why the draft is not helpful: {helpfulness_classification.explanation}'

def get_helpfulness_classification_str(helpfulness_classification: HelpfulnessClassification) -> str:
    return f'\nHelpfulness verdict: {repr(helpfulness_classification.category)}\nHelpfulness explanation: {helpfulness_classification.explanation}'

def get_response_draft_str(response_draft: ResponseDraft) -> str:
    return f'\nResponseDraft: {response_draft.responseDraft}\nResponseDraft explanation: {response_draft.explanation}'

print(get_response_draft_str(response_draft))

# exemplary use of helpfulness_classifier_chain:
helpfulness_classification = helpfulness_classifier_chain.invoke({
    "messages": get_messages_str(messages),
    "chunks": get_chunks_str(chunks),
    "missing_info": get_missing_info_str(missing_customer_info),
    "response_draft": get_response_draft_str(response_draft),
})
print(helpfulness_classification)

#%%
print("# Manual intervention detection")

class InterventionClasses(str, Enum):
    INTERVENTION = "human needed"
    NO_INTERVENTION = "no human needed"
class InterventionClassification(BaseModel):
    explanation: str = Field(description="An explanation for the classification result")
    category: InterventionClasses = Field(description="The classification result")
structured_llm = llm.with_structured_output(InterventionClassification, method="json_schema")

manual_intervention_detection_sys_msg = ("system", customer_service_sys_msg_prefix + classification_sys_msg_prefix + """
   Conduct a comprehensive step by step analysis for the classification, using the data provided below:
   * a list of messages, showing the conversation between MediVoice and the customer so far
   * chunks from MediVoice's vector database that are related to the last customer message
   * missing information detection result, showing whether the customer response should aks the customer for missing information
   * a draft for a reply to the customer, together with a reasoning why the draft is not helpful.
   Then classify the provided data into one of the following given classes:
    * "human needed": a MediVoice employee needs to get involved to get helpful messages to make progress towards solving the customer's problem
    * "no human needed": based on the reasoning why the draft is not helpful, the AI will be able to generate a reply to the customer that will make progress towards solving the customer's problem.
    Output nothing but one of the classes, i.e. "human needed" or "no human needed", nothing else.
    """)
manual_intervention_detection_prompt = ChatPromptTemplate.from_messages([manual_intervention_detection_sys_msg, ("user", "{messages}\n{chunks}\n{missing_info}\n{response_draft}\n{unhelpfulness_reason}")])
intervention_classifier_chain = missing_info_detection_prompt | structured_llm

# exemplary use of intervention_classifier_chain:
print(intervention_classifier_chain.invoke({
    "messages": get_messages_str(messages),
    "chunks": get_chunks_str(chunks),
    "missing_info": get_missing_info_str(missing_customer_info),
    "response_draft": get_response_draft_str(response_draft),
    "unhelpfulness_reason": get_unhelpful_reason_str(helpfulness_classification),
}))

#%%
print("# Issue tracker item detection TODO")


#%%
"""## State"""

from langchain.schema import Document
from langgraph.graph import END, StateGraph

from typing_extensions import TypedDict
from typing import List

### State

class GraphState(TypedDict):
    messages : List[str] # each is a customer message or MediVoice response
    alternativeQueries: AlternativeQueries
    chunks: Annotated[List[str], TOP_K_CHUNKS]
    enoughCustomerInfo: CustomerInfoClassification
    num_steps: Annotated[int, TOP_K_CHUNKS]
    draft_email_feedback : dict

"""## Nodes

1. categorize_message
2. rag  
3. draft_message_writer  
4. analyze_draft_message  
5. rewrite_message  
6. no_rewrite  
7. state_printer


0. read initial customer message, loop_counter = 0
conditional edge with expectation detection: expectation_classifier_chain
1. query_rewriting_chain
2. rag_chain
3. missing_info_classifier_chain
4. response_draft_chain
conditional edge with helpfulness detection: helpfulness_classifier_chain
conditional edge with manual intervention detection: intervention_classifier_chain
5. increase loop_counter
6. issue_tracker_classifier_chain
7. send message, loop_counter = 0
8. create issue tracker item
9. fill issue tracker item
"""

def categorize_message_expectation(state):
    """take the initial customer message and categorize it"""
    print("# Categorize message expectation")

    # get the state
    # invoke the chain
    # print
    # save to local disk
    return {"message_category": message_category}

"""## Conditional Edges"""

def route_to_response_draft_generation(state):
    """
    Route back to draft a response to teh customer.
    """

    # get state
    # invoke the classification chain
    # print classification result
    if result.category == "response needed":
        print("route to response draft generation")
        return "draft_response_node"
    else: 
        print("route to send message")
        return "send_message_node"

"""## Build the Graph

### Add Nodes
"""

graph = StateGraph(GraphState)

graph.add_node("categorize_message", categorize message) 
# TODO ...

"""### Add Edges"""

graph.set_entry_point("initial_customer_message")

graph.add_conditional_edges(
    "response needed",
    route_to_response_draft_generation,
    {
        "route to response draft generation": "draft_response_node",
        "route to send message": "send_message_node",
    },
)
graph.add_edge("send_message_node", "issue_tracker_node")

app = graph.compile()
inputs = {"initial_message": initial_customer_messages[0],"draft": None}
for output in app.stream(inputs):
    for key, value in output.items():
        print(f"Finished running: {key}:")
print(app.invoke(inputs))
