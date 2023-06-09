from langchain.agents import (Tool, AgentExecutor, LLMSingleActionAgent,
                              AgentOutputParser, AgentType, create_json_agent, create_openapi_agent, initialize_agent)
from langchain.prompts import StringPromptTemplate
from langchain import LLMChain, SerpAPIWrapper, WikipediaAPIWrapper
from langchain.chat_models import ChatOpenAI
from typing import List, Union, Callable
from langchain.schema import AgentAction, AgentFinish, Document
from langchain.requests import RequestsWrapper
import os
import re
import yaml
from langchain.agents.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain.agents.agent_toolkits import OpenAPIToolkit, JsonToolkit
from langchain.agents.agent_toolkits.openapi import planner
import spotipy.util as util
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools.json.tool import JsonSpec

with open("spotify_openapi.yaml") as f:
    raw_spotify_api_spec = yaml.load(f, Loader=yaml.Loader)
spotify_api_spec = reduce_openapi_spec(raw_spotify_api_spec)
json_spec = JsonSpec(dict_=raw_spotify_api_spec, max_value_length=4000)
# spotify_toolkit = JsonToolkit(spec=json_spec)


def construct_spotify_auth_headers(raw_spec: dict):
    scopes = list(raw_spec['components']['securitySchemes']
                  ['oauth_2_0']['flows']['authorizationCode']['scopes'].keys())
    access_token = util.prompt_for_user_token(scope=','.join(scopes))
    # print(access_token)
    return {
        'Authorization': f'Bearer {access_token}'
    }


# Get API credentials.
headers = construct_spotify_auth_headers(raw_spotify_api_spec)
spotify_requests_wrapper = RequestsWrapper(headers=headers)

# openapi_toolkit = OpenAPIToolkit.from_llm(ChatOpenAI(
#     model_name="gpt-3.5-turbo", temperature=0.1), json_spec, spotify_requests_wrapper)

openapi_agent_executor = planner.create_openapi_agent(spotify_api_spec, spotify_requests_wrapper,
                                                      llm=ChatOpenAI(
                                                          temperature=0, verbose=True)
                                                      )
# spotify_agent_executor = create_json_agent(llm=ChatOpenAI(
#     temperature=0.1), toolkit=spotify_toolkit, verbose=True)

# Define which tools the agent can use to answer user queries
wiki_api = WikipediaAPIWrapper()
tools = [
    Tool(
        name="Wikipedia",
        func=wiki_api.run,
        description="Useful for when you need to answer general questions about people, places, companies, historical events"),
    Tool(
        name="Spotify",
        func=openapi_agent_executor.run,
        description="useful when you have query related to songs"
    )
]
# [
#     Tool(
#     name="Search",
#     func=search_api.run,
#     description="useful for when you need to answer questions about current events",
# ),
# Tool(
#     name="Wikipedia",
#     func=wiki_api.run,
#     description="Useful for when you need to answer general questions about people, places, companies, historical events"
# )
# ]
# spotify_tool = Tool(
#         name = "Spotify",
#         func=spotify_requests_wrapper.get,
#         description="useful when you have query related to songs"
#     )

docs = [Document(page_content=t.description, metadata={
                 "index": i}) for i, t in enumerate(tools)]
vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
retriever = vector_store.as_retriever()


def get_tools(query):
    docs = retriever.get_relevant_documents(query)
    return [tools[d.metadata["index"]] for d in docs]


# Set up the base template
template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}"""

# Set up a prompt template


class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    tools_getter: Callable

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        ############## NEW ######################
        tools = self.tools_getter(kwargs["input"])
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
        return self.template.format(**kwargs)


prompt = CustomPromptTemplate(
    template=template,
    tools_getter=get_tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
)


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(

                return_values={"output": llm_output.split(
                    "Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


output_parser = CustomOutputParser()

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, verbose=True)
# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names
)

# agent_executor = AgentExecutor.from_agent_and_tools(
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, tools=tools, verbose=True)

multi_agent = initialize_agent(
    llm=llm, tools=tools, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

multi_agent.run("What are 2 famous Taylor Swift songs?")
