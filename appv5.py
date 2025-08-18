# updated version for app.py: combine langgraph to create workflow for process pdf and persistence memory

import os # module to interact with the operating system

import deepl
from dotenv import load_dotenv # module to load environment variables from a .env file
from langchain_groq import ChatGroq
import chainlit as cl

from utils.stormv2 import run_storm_pipeline
from utils.parse import parse_all_markdown_sections

from langchain.callbacks.tracers import LangChainTracer
import asyncio
from langsmith import Client
import json
import re
from typing import List, Tuple, Annotated, Any
from langgraph.graph import add_messages
from langchain.agents import initialize_agent, Tool
from langchain_core.tools import StructuredTool
import tracemalloc
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing_extensions import Literal
from typing import TypedDict, List
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.prebuilt import ToolNode
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import PyPDFLoader
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain.chains import RetrievalQA
from langchain_core.documents import Document

tracemalloc.start()  # Start tracing memory allocations

load_dotenv()

auth = os.getenv("DEEPL_API_KEY")

DB_URI = "postgresql://localhost:5432/anhtuantran?sslmode=disable"

if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

if not os.environ.get("MISTRAL_API_KEY"):
    os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY")

if not os.environ.get("GROQ_API_KEY"):
  os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

model = ChatGroq(model="llama-3.3-70b-versatile")


embeddings = MistralAIEmbeddings(model="mistral-embed")
vector_store = Chroma(
    collection_name="pdf_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"
)



class State(TypedDict):
#workflow define state
    messages: Annotated[List[BaseMessage], add_messages]
    input: str
    decision: str
    output_question: str
    output_essay: {str, list[tuple[str, int]]}
    pdf_path: str
    pdf_text: str
    context: str
    retriever: Any


################################################################
#workflow 


#tools generate essay
def run_storm_pipeline_wrapper(msg: str):

    topic = msg

    run_storm_pipeline(
        topic= topic,
        retriever="tavily",
        output_dir="./data",
        do_research=True,
        do_generate_outline=True,
        do_generate_content=True,
        do_polish_article=True
        )

    result_path = f"./data/{topic.replace(' ', '_').replace('/', '_')}/storm_gen_article.txt"
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            article = f.read()

    url_info_path = f"./data/{topic.replace(' ', '_').replace('/', '_')}/url_to_info.json"

    with open(url_info_path, "r") as f:
        full_info = json.load(f)

    # kiểm tra file link url 
    url_map = full_info.get("url_to_unified_index", {})


    sorted_urls = sorted(url_map.items(), key=lambda x: x[1])

    return {
        "article_translated": article,
        "sorted_urls": sorted_urls
    }
async def arun_storm_pipeline_wrapper(msg: str):

    topic = msg

    await asyncio.to_thread(run_storm_pipeline,
        topic= topic,
        retriever="tavily",
        output_dir="./data",
        do_research=True,
        do_generate_outline=True,
        do_generate_article=True,
        do_polish_article=True
        )

    result_path = f"./data/{topic.replace(' ', '_').replace('/', '_')}/storm_gen_article.txt"
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            article = f.read()

    url_info_path = f"./data/{topic.replace(' ', '_').replace('/', '_')}/url_to_info.json"

    with open(url_info_path, "r") as f:
        full_info = json.load(f)

    # kiểm tra file link url 
    url_map = full_info.get("url_to_unified_index", {})


    # Sort by index (value)
    sorted_urls = sorted(url_map.items(), key=lambda x: x[1])

    return {
        "article_translated": article,
        "sorted_urls": sorted_urls
    }
run_pipeline = StructuredTool.from_function(func=run_storm_pipeline_wrapper,
                                            coroutine=arun_storm_pipeline_wrapper,
                                            description="write essay article with provided topic and requirements",
                                            return_direct=True
                                            )

#tools answer normal question
def call_llm_model(prompt: str, messages: List) -> str:


    response = model.invoke(
        [
            SystemMessage(
                content="""
                You are a highly intelligent, versatile assistant with deep expertise across academic, technical, and professional domains. Your task is to help users by understanding their intent, retrieving or generating accurate information, and presenting it in a clear, structured, and context-appropriate format. If you do not know something, say I do not know. Do not fabricate information or invent sources.
                """
            ),
            # *messages,
            # HumanMessage(
            #     content=prompt
            # ),
            *messages
        ]
        )
    return {"messages" : response}
async def acall_llm_model(prompt: str, messages: List) -> str:


    response = await model.ainvoke(
        [
            SystemMessage(
                content="""
                You are a highly intelligent, versatile assistant with deep expertise across academic, technical, and professional domains. Your task is to help users by understanding their intent, retrieving or generating accurate information, and presenting it in a clear, structured, and context-appropriate format. If you do not know something, say I do not know. Do not fabricate information or invent sources.
                """
            ),
            *messages
        ]
        )
    return {"messages" : response}
answer_tool = StructuredTool.from_function(func=call_llm_model,
                                           coroutine=acall_llm_model,
                                           description="Tool that is called when the user asks a question or requests information that requires the LLM model's capabilities",
                                           return_direct=True
                                        )

tool_node = ToolNode([answer_tool])


class Route(BaseModel):
    step: Literal["write_essay", "answer_question"] = Field(
        None, description="the next step in the routing process"
    )

router = model.with_structured_output(Route)

async def llm_call_router(state: State):
    """Route the input to the appropriate node"""
    decision = await router.ainvoke(
        [
            SystemMessage(
                content="""
                You are a routing model.
                Your task: return ONLY one of these exact strings, without quotes, and nothing else:
                - write_essay
                - answer_question
                Do not return synonyms, abbreviations, or variations.
                """
            ),
            HumanMessage(
                content=state["input"]
            )
        ]
    )
    return {"decision": decision.step}

async def pre_llm_call_essay(state: State):
    """Preprocess the input for essay generation"""
    response = await model.ainvoke(
        [
            SystemMessage(
                content="""
                You are a helpful assistant who helps to summarize the message input and generate a concise prompt for generating essay step. Remember just take those messages that are relevant and remember to keep the prompt short, ideally under 30 words.
                """
            ),
            *state["messages"],
        ]
    )
    return {"input": response.content}


async def llm_call_essay(state: State):
    """Call the essay generation tool, only called when the user explicitly says a word that means 'write an essay'"""
    result = await run_pipeline.ainvoke({"msg": state["input"]})

    return {
        "output_essay": {
            "article_translated": result["article_translated"],
            "sorted_urls": result["sorted_urls"]
        }
    }

async def pre_llm_call_answer(state: State):
    """Preprocess the input for answer generation"""
    retriever = vector_store.as_retriever()
    llm = init_chat_model("llama3-8b-8192", model_provider="groq")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    response = await qa_chain.ainvoke({"query": state["input"]})
    print("😎", response["result"])
    return {
        "messages": [SystemMessage(content=response["result"])],
        "context": response["result"]
    }


async def llm_call_answer(state: State):
    """answer short question or request based on the general knowledge of the LLM model"""
    res = await answer_tool.ainvoke({"prompt": state["context"], "messages": state["messages"]})
    resp = res["messages"]
    return {
        "messages": [resp]
    }

async def router_decision(state: State):
    if state["decision"] == "write_essay":
        return "pre_llm_call_essay"
    elif state["decision"] == "answer_question":
        return "pre_llm_call_answer"
    else:
        raise ValueError(f"Unknown decision: {state['decision']}")

async def tools(state: State):
    message = state["messages"]
    last_message = message[-1]
    if last_message.tool_calls:
        tool_result = await tool_node.ainvoke({"messages": [last_message]})
        return {
            "output_question": tool_result["messages"][0].content,
            "messages": state["messages"] + [last_message] + tool_result["messages"]
        }
    else:
        return {
            "output_question": last_message.content,
            "messages": state["messages"] + [last_message]
        }

router_builder = StateGraph(State)

router_builder.add_node("pre_llm_call_essay", pre_llm_call_essay)
router_builder.add_node("pre_llm_call_answer", pre_llm_call_answer)
router_builder.add_node("llm_call_essay", llm_call_essay)
router_builder.add_node("llm_call_answer", llm_call_answer)
router_builder.add_node("llm_call_router", llm_call_router)
router_builder.add_node("tools", tools)

router_builder.add_edge(START, "llm_call_router")
router_builder.add_conditional_edges(
    "llm_call_router",
    router_decision,
    {
        "pre_llm_call_essay": "pre_llm_call_essay",
        "pre_llm_call_answer": "pre_llm_call_answer"
    },
)
router_builder.add_edge("pre_llm_call_essay", "llm_call_essay")
router_builder.add_edge("pre_llm_call_answer", "llm_call_answer")
router_builder.add_edge("llm_call_essay", END)
router_builder.add_edge("llm_call_answer", "tools")
router_builder.add_edge("tools", END)
memory = MemorySaver()


router_workflow = router_builder.compile(checkpointer=memory)

##############################################################

##############################################################
# pdf loader node
async def pdf_loader_node(state: State):
    loader = PyPDFLoader(state["pdf_path"])
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    _ = vector_store.add_documents(splits)
    state["pdf_text"] = " ".join([doc.page_content for doc in splits])
    # state["retriever"] = vector_store.as_retriever()
    return state



##############################################################
# message handler chainlit
@cl.on_message
async def handle_message(msg: cl.Message):

    config: RunnableConfig = { 'configurable' : {'thread_id' : cl.context.session.thread_id} }

    state_update = {}
    ##################################
    # pdf workflow
    if msg.elements:
        print("👃🏻")
        for element in msg.elements:
            #TEMP
            if element.mime == "application/pdf":
                result = await pdf_loader_node({"pdf_path": element.path})
                print("📄 PDF loaded and processed successfully.", result["pdf_text"])

    ##################################
    # text workflow
    elif msg.content:

        result = await router_workflow.ainvoke({"input": msg.content, 'messages': [HumanMessage(msg.content)]}, config=config)
        print("🧪 result:", result)  # debug
        if "output_essay" in result:
            if result["output_essay"] is not None:
                if "article_translated" in result["output_essay"]:
                    article_translated = result["output_essay"]["article_translated"]
                    print("🧪 clear:", result["output_question"])
                    sorted_urls = result["output_essay"]["sorted_urls"]
                    await cl.Message(content=f"✅ Article generated:\n\n{article_translated}").send()
                    state_update["output_essay"] = None
                    state_update["decision"] = None
                    markdown_links = "\n".join(
                        f"{i+1}. [{url}]({url})" for i, (url, _) in enumerate(sorted_urls)
                    )
                    await cl.Message(content="🔗 **Sources used in article (from `url_to_info.json`)**:\n" + markdown_links).send()
        if "output_question" in result:
            if result["output_question"] is not None:
                messages = result["output_question"]

                print("🧪 output_question messages:", messages)
                state_update["output_question"] = None
                state_update["decision"] = None

                await cl.Message(content=f"Answer:\n\n{messages}").send()

        if state_update:
            router_workflow.update_state(config, state_update)
            print("🧹 Cleared from persistent state:", state_update)