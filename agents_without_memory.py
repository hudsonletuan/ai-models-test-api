# You need to install "langchain" and "langchain-openai" and "openai" and "duckduckgo-search" 
# and "langchainhub" and "langchain-community tavily-python" and "faiss-gpu" packages first to have access to utilities
# to Langchain and use OpenAI and Tavily API key

import getpass
import os

os.environ["TAVILY_API_KEY"] = getpass.getpass()

from langchain_community.tools.tavily_search import TavilySearchResults

search = TavilySearchResults()
search.invoke("what is the most uptrending stock today?")

from langchain.agents import AgentExecutor, AgentType, load_tools, initialize_agent
from langchain.chat_models import ChatOpenAI

from getpass import getpass

OPENAI_API_KEY = getpass()

import os

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)
vector = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vector.as_retriever()

retriever.get_relevant_documents("how to upload a dataset")[0]

from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)

tools = [search, retriever_tool]

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

from langchain import hub

prompt = hub.pull("hwchase17/openai-functions-agent")
prompt.messages

from langchain.agents import create_openai_functions_agent

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "hi!"})
agent_executor.invoke({"input": "what is an API key?"})
agent_executor.invoke({"input": "how to create an ASP.NET Core environment in VSCode?"})
agent_executor.invoke({"input": "what is the most popular programming language in 2023?"})

# Another method

def build_simple_agent():
  llm = ChatOpenAI(openai_api_base="https://api.openai.com/v1",
                   api_key=OPENAI_API_KEY,
                   model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
                   temperature=0.7,
                   verbose=True)
  tools = load_tools(["llm-math", "ddg-search"], llm=llm)
  agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
  print(agent.agent.llm_chain.prompt.template)
  return agent

def chat_with_agent(agent: AgentExecutor, user_input: str):
  output = agent.invoke({"input": user_input})
  return output

agent = build_simple_agent()
agent.invoke({"input": "what is an API key?"})