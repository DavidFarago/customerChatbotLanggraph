"""# Customer Chatbot Flow

Respond to new customer message:
1. wait for new customer message. Upon new customer message or timeout, go to 2.
2. expectation detection based on messages:
  * "response needed"
  * "no response needed".

If "no response needed", go to 10, else go to 3.
3. use last customer message (and last queries/keywords) to create queries/keywords for query rewriting RAG
4. retrieve chunks based on last customer message and last messages and created queries or keywords
5. missing info detection based on the data:
  * "needs further customer info",
  * "sufficient customer info".
6. use messages, retrieved chunks and missing info detection output (and possibly previous draft and helpfulness critique) to write a draft reply
7. check draft reply for helpfulness: is the draft reply helpfull in moving the problem forward (e.g. by answering the customer inquery or by asking the customer for further information that will help move the problem forward)?
  * No: store helpfulness critique and goto 8.
  * Yes: go to 9.
8. Manual intervention detection based on the data:
  * "human needed"
  * "no human needed"

If human needed or loop counter > RETRIES: go to 10.
Else increase loop counter and go to 6.
9. Respond to the customer with the generated message and go to 1 
10. issue tracker item detection:
  * "issue tracker item needed": there should be an issue tracker item (e.g. because a MediVoice employer needs to get involved), but is not yet
  * "no issue tracker item needed": it is not worth to create an issue tracker item for these messages, or an issue tracker tem is already present in the messages.

If issue tracker item needed, goto 11, else goto 12.
11. Create an issue tracker item, send the customer a reply with the item ID, and go to 12.
12. If there is an issue item: {store all information in the issue item}. Terminate.
"""


#%%
from dotenv import load_dotenv
from enum import Enum
import operator
import os
from pydantic import BaseModel, Field
from typing import Annotated, List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.globals import set_debug

load_dotenv()

def get_config_value(key: str, default, cast_func=str):
    raw_value = os.getenv(key)
    if raw_value is None:
        result = default
    else:
        try:
            result = cast_func(raw_value)
        except Exception as e:
            raise ValueError(f"Invalid value for {key}: {raw_value}. Error: {e}")
    print(f"Value of {key}: {result}")
    return result

str_to_bool = lambda s: s.lower() in ['true', '1', 'yes', 'y']

# hyper parameters:
TOP_K_FUSION_QUERIES = get_config_value("TOP_K_FUSION_QUERIES", 3, int)
TOP_K_CHUNKS = get_config_value("TOP_K_CHUNKS", 3, int)
CHUNK_SIZE = get_config_value("CHUNK_SIZE", 1000, int) 
CHUNK_OVERLAP = get_config_value("CHUNK_OVERLAP", 200, int)
RETRIES = get_config_value("RETRIES", 3, int)

# tracing:
DEBUG = get_config_value("DEBUG", False, str_to_bool)
LANGSMITH_TRACING    = get_config_value("LANGSMITH_TRACING", DEBUG, str_to_bool)
LANGSMITH_ENDPOINT   = get_config_value("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com", str)
LANGSMITH_API_KEY = get_config_value("LANGSMITH_API_KEY", "get_it_at_smith.langchain.com", str)
LANGSMITH_PROJECT = "customer_chatbot"
LANGCHAIN_TRACING_V2 = get_config_value("LANGCHAIN_TRACING_V2", DEBUG, str_to_bool)
LANGCHAIN_ENDPOINT = get_config_value("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com", str)
LANGCHAIN_API_KEY = get_config_value("LANGCHAIN_API_KEY", LANGSMITH_API_KEY, str)
LANGGRAPH_STUDIO = get_config_value("LANGGRAPH_STUDIO", False, str_to_bool)

set_debug(DEBUG)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

initial_customer_messages = [
    "Why is my phone assistant not offering any appointments in Febuary?",
    # "I used the default settings, so I think this is a bug in your system. Please investigate and get back to me!"
    # "Never mind, I made a mistake. Sorry for the confusion."
    "How do I configure multiple locations for my practice?",
    "Wie kann ich mehrere Standorte für meine Praxis konfigurieren?",
    "Can I configure my assistant to understand Spanish besides German?",
    "Kann ich meinen Assistenten so konfigurieren, dass er neben Deutsch auch Spanisch versteht?"
    "Your system is great, keep up the good work!",
    "We stopped using your assistant and canceled our contract with MediVoice. Goodbye.",
    "I have a bug report: Your assistant does not schedule appointments on Sundays, even though I configured slots for Sundays.",
    "Ich habe einen Fehler zu melden: Der Assistent vergibt keine Termine an Sonntagen, obwohl ich Slots für Sonntage konfiguriert habe.",
    "I get back different names for the same person, in the same call :-0",
    "Ich bekomme unterschiedliche Namen für die gleiche Person, im selben Anruf :-0"
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
    NEEDS_RESPONSE = "response needed"
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
messages = [f"Customer message: {initial_customer_messages[-1]}"] # part of graph state
print(f"Starting off with customer message: {initial_customer_messages[-1]}")
print(expectation_classifier_chain.invoke({"messages": get_messages_str(messages)}))


#%%
print("# Create queries or keywords")

class AlternativeQueries(BaseModel):
    explanation: str = Field(description="An explanation for all the alternative queries/keywords")
    queries: Annotated[List[str], TOP_K_FUSION_QUERIES] = Field(description="The alternative queries/keywords")
structured_llm = llm.with_structured_output(AlternativeQueries, method="json_schema")

create_queries_keywords_sys_msg = ("system", customer_service_sys_msg_prefix + """
    You are a RAG fusion expert, i.e. a master at creating alternative queries/keywords that are related to the information you will be provided: a customer message and possibly earlier alternative queries/keywords.
    Create {number_queries} alternative queries/keywords that rephrase the provided customer message to perfect the retrieval of chunks relevant to the provided customer message from a vector database.

    First think step by step to work out the best {number_queries} alternative queries/keywords that are closely related to the customer message (and possibly earlier alternative queries/keywords) provided below, to increase the relevance of the retrieved chunks for the provided customer message.
    Each alternative can be a query or a keyword, whatever you think is better. The provided customer message and each of your {number_queries} generated alternative queries/keywords will be used separately to retrieve relevant chunks from a vector database.

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
if TOP_K_FUSION_QUERIES  == 0:
    print("No queries needed")
    alternative_queries = AlternativeQueries(explanation="", queries=[]) # part of graph state
else:    
    alternative_queries = alternative_queries_chain.invoke({
        "number_queries": TOP_K_FUSION_QUERIES,
        "last_customer_message": get_last_customer_message_str(messages),
        "previous_queries": get_optional_previous_queries_str(previous_queries),
    }) # part of graph state
    print(f"Alternative {TOP_K_FUSION_QUERIES} queries:")
    print(alternative_queries)


#%%
print("# Retrieve chunks")

from langchain_community.document_loaders import UnstructuredMarkdownLoader
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
import shutil
persist_directory = 'db'
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)
vectordb = Chroma.from_documents(documents=texts,
                                 embedding=openai_embeddings,
                                 persist_directory=persist_directory)
#like retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K_CHUNKS}), but includes similarity scores:
from langchain_core.documents import Document
from langchain_core.runnables import chain
@chain
def retriever(query: str) -> List[Document]:
    docs, scores = zip(*vectordb.similarity_search_with_score(query, k=TOP_K_CHUNKS))
    for doc, score in zip(docs, scores):
        doc.metadata["score"] = score
    return docs

def get_unique_sorted_documents(docs):
    unique = {}
    for doc in docs:
        content = doc.page_content
        score = doc.metadata.get("score", 0)
        if content not in unique or score > unique[content].metadata.get("score", 0):
            unique[content] = doc
    sorted_docs = sorted(unique.values(), key=lambda d: d.metadata.get("score", 0), reverse=True)
    return [doc.page_content for doc in sorted_docs]

# exemplary use of chunk:
chunks_collector = []
for query in [get_last_customer_message_str(messages)] + alternative_queries.queries:
    new_chunks = retriever.invoke(query[18:])
    chunks_collector.extend(new_chunks)
print(f"Retrieved {len(chunks_collector)} chunks in total: {chunks_collector}")
chunks: ChunkArray = get_unique_sorted_documents(chunks_collector) # part of graph state
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
    if not helpfulness_classification or helpfulness_classification.category == HelpfulnessClasses.HELPFUL:
        return ''
    return f'\nReasoning why the draft is not helpful: {helpfulness_classification.explanation}'

def get_helpfulness_classification_str(helpfulness_classification: HelpfulnessClassification) -> str:
    return f'\nHelpfulness verdict: {repr(helpfulness_classification.category)}\nHelpfulness explanation: {helpfulness_classification.explanation}'

def get_response_draft_str(response_draft: ResponseDraft) -> str:
    if not response_draft:
        return ''
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
print("\n# Building and executing the langGraph\n")

# TODO: syntactically simplify conditional edges

# TODO: Build this as an agent, as in AI-Agents-In-LangGraph course

# TODO: Use ConversationChain for better modularity? 

# The langGraph state
from langgraph.graph import END, StateGraph
from langgraph.errors import NodeInterrupt
from typing_extensions import TypedDict

class GraphState(TypedDict):
    messages : Annotated[list[str], operator.add] # each is a customer message or MediVoice response
    expectation_classification: ExpectationClassification
    alternative_queries: AlternativeQueries
    chunks: ChunkArray
    missing_customer_info: CustomerInfoClassification
    response_draft: ResponseDraft
    helpfulness_classification: HelpfulnessClassification
    intervention_classification: InterventionClassification
    tries: Annotated[int, RETRIES]
graph = StateGraph(GraphState)

# The langGraph nodes and edges
"""
0. receive_customer_message
conditional edge with expectation detection: expectation_classifier_chain
1. rewrite_query
2. get_chunks
3. classify_missing_info
4. draft_response
conditional edge with helpfulness detection: helpfulness_classifier_chain
conditional edge with manual intervention detection: intervention_classifier_chain
5. send_message
6. increase loop_counter
7. issue_tracker_classifier
8. send message, loop_counter = 0
9. create issue tracker item
10. fill issue tracker item
"""

def receive_customer_message(_state: GraphState):
    if LANGGRAPH_STUDIO:
        return {} # message will be appended to `messages` in the studio
    customer_message = input("Receive customer message: ")
    return {"messages": [f"Customer message: {customer_message}"], "tries": 0}

graph.add_node("receive_customer_message", receive_customer_message) 
graph.set_entry_point("receive_customer_message")

def classify_expectation(state: GraphState):
    expectation_classification_result = expectation_classifier_chain.invoke({"messages": get_messages_str(state["messages"])})
    print(f"Classified expectation: {expectation_classification_result.category}")
    return {"expectation_classification": expectation_classification_result}

graph.add_node("classify_expectation", classify_expectation)
graph.add_edge("receive_customer_message", "classify_expectation")

graph.add_conditional_edges(
    "classify_expectation",
    lambda state: "classify_issue_tracker_item" if state["expectation_classification"].category == ExpectationClasses.NO_RESPONSE else "rewrite_query",
    {
        "classify_issue_tracker_item": "classify_issue_tracker_item",
        "rewrite_query": "rewrite_query",
    },
)

def rewrite_query(state: GraphState):
    if TOP_K_FUSION_QUERIES == 0:
        print("No queries needed")
        return {"alternative_queries": None}
    alternative_queries = alternative_queries_chain.invoke({
        "number_queries": TOP_K_FUSION_QUERIES,
        "last_customer_message": get_last_customer_message_str(state["messages"]),
        "previous_queries": get_optional_previous_queries_str(state.get("alternative_queries")),
    })
    print(f"Rewritten queries: {alternative_queries.queries}")
    return {"alternative_queries": alternative_queries}

graph.add_node("rewrite_query", rewrite_query)

def get_chunks(state: GraphState):
    chunks_collector = []
    for query in [get_last_customer_message_str(state["messages"])] + state["alternative_queries"].queries:
        chunks_collector.extend(retriever.invoke(query))
    chunks: ChunkArray = {chunk.page_content for chunk in chunks_collector}
    print(f"Got {len(chunks)} chunks")
    return {"chunks": chunks}

graph.add_node("get_chunks", get_chunks)
graph.add_edge("rewrite_query", "get_chunks")

def classify_missing_info(state: GraphState):
    missing_customer_info = missing_info_classifier_chain.invoke({
        "messages": get_messages_str(state["messages"]),
        "chunks": get_chunks_str(state["chunks"])
    })
    print(f"Classified missing info: {missing_customer_info.category}")
    return {"missing_customer_info": missing_customer_info}

graph.add_node("classify_missing_info", classify_missing_info)
graph.add_edge("get_chunks", "classify_missing_info")

def draft_response(state: GraphState):
    previous_response_draft_or_empty = get_response_draft_str(state.get("response_draft"))
    previous_unhelpfulness_reason_or_empty = get_unhelpful_reason_str(state.get("helpfulness_classification"))
    response_draft = response_draft_chain.invoke({
        "messages": get_messages_str(state["messages"]),
        "chunks": get_chunks_str(state["chunks"]),
        "missing_info": get_missing_info_str(state["missing_customer_info"]),
        "response_draft": previous_response_draft_or_empty,
        "unhelpfulness_reason": previous_unhelpfulness_reason_or_empty,
    })
    print(f"Drafted response: {response_draft.responseDraft}")
    return {"response_draft": response_draft, "tries": state["tries"] + 1}

graph.add_node("draft_response", draft_response)
graph.add_edge("classify_missing_info", "draft_response")

def classify_helpfulness(state: GraphState):
    helpfulness_classification_result = helpfulness_classifier_chain.invoke({
        "messages": get_messages_str(state["messages"]),
        "chunks": get_chunks_str(state["chunks"]),
        "missing_info": get_missing_info_str(state["missing_customer_info"]),
        "response_draft": get_response_draft_str(state["response_draft"]),
    })
    print(f"Classified helpfulness: {helpfulness_classification_result.category}")
    return {"helpfulness_classification": helpfulness_classification_result}

graph.add_node("classify_helpfulness", classify_helpfulness)
graph.add_edge("draft_response", "classify_helpfulness")

graph.add_conditional_edges(
    "classify_helpfulness",
    lambda state: "classify_manual_intervention" if state["helpfulness_classification"].category == HelpfulnessClasses.NOT_HELPFUL else "send_message",
    {
        "classify_manual_intervention": "classify_manual_intervention",
        "send_message": "send_message",
    },
)

def send_message(state: GraphState):
    print(f'Sent the customer the message "{state["response_draft"].responseDraft}"')
    return {"messages": [f"MediVoice response: {state['response_draft'].responseDraft}"]}

graph.add_node("send_message", send_message)
graph.add_edge("send_message", "receive_customer_message")

def classify_manual_intervention(state: GraphState):
    intervention_classification_result = intervention_classifier_chain.invoke({
        "messages": get_messages_str(state["messages"]),
        "chunks": get_chunks_str(state["chunks"]),
        "missing_info": get_missing_info_str(state["missing_customer_info"]),
        "response_draft": get_response_draft_str(state["response_draft"]),
        "unhelpfulness_reason": get_unhelpful_reason_str(state["helpfulness_classification"]),
    })
    print(f"Classified intervention: {intervention_classification_result.category}")
    return {"intervention_classification": intervention_classification_result}

graph.add_node("classify_manual_intervention", classify_manual_intervention)

graph.add_conditional_edges(
    "classify_manual_intervention",
    lambda state: "classify_issue_tracker_item" if state["intervention_classification"].category == InterventionClasses.INTERVENTION or state["tries"] > RETRIES else "draft_response",
    {
        "classify_issue_tracker_item": "classify_issue_tracker_item",
        "draft_response": "draft_response",
    },
)

def classify_issue_tracker_item(state: GraphState):
    print("""Decide issue tracker item workflow not implemented yet:
10. issue tracker item detection:
  * "issue tracker item needed": there should be an issue tracker item (e.g. because a MediVoice employer needs to get involved), but is not yet
  * "no issue tracker item needed": it is not worth to create an issue tracker item for these messages, or an issue tracker tem is already present in the messages.

If issue tracker item needed, goto 11, else goto 12.
11. Create an issue tracker item, send the customer a reply with the item ID, and go to 12.
12. If there is an issue item: {store all information in the issue item}. Terminate.""")
    return {}
graph.add_node("classify_issue_tracker_item", classify_issue_tracker_item)
graph.add_edge("classify_issue_tracker_item", END)

if LANGGRAPH_STUDIO:
    graph = graph.compile(interrupt_before=['receive_customer_message'])
else: 
    graph = graph.compile()
    graph.get_graph().draw_png('graph.png')
    graph.get_graph().draw_mermaid_png(output_file_path='graph_mermaid.png')
    graph.invoke({"messages": [], "tries": 0})
