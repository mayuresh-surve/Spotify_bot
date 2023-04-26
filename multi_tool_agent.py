from langchain.agents import (Tool, AgentType, create_json_agent, create_openapi_agent, initialize_agent)
from langchain import WikipediaAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.requests import RequestsWrapper
import os
import re
import yaml
from langchain.agents.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain.agents.agent_toolkits.openapi import planner
import spotipy.util as util
from langchain.tools.json.tool import JsonSpec

with open("spotify_openapi.yaml") as f:
    raw_spotify_api_spec = yaml.load(f, Loader=yaml.Loader)
spotify_api_spec = reduce_openapi_spec(raw_spotify_api_spec)
json_spec = JsonSpec(dict_=raw_spotify_api_spec, max_value_length=4000)


def construct_spotify_auth_headers(raw_spec: dict):
    scopes = list(raw_spec['components']['securitySchemes']
                  ['oauth_2_0']['flows']['authorizationCode']['scopes'].keys())
    access_token = util.prompt_for_user_token(scope=','.join(scopes))
    return {
        'Authorization': f'Bearer {access_token}'
    }

# Get API credentials.
headers = construct_spotify_auth_headers(raw_spotify_api_spec)
spotify_requests_wrapper = RequestsWrapper(headers=headers)

openapi_agent_executor = planner.create_openapi_agent(spotify_api_spec, spotify_requests_wrapper,
                                                      llm=ChatOpenAI(
                                                          temperature=0, verbose=True)
)
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
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, verbose=True)

multi_agent = initialize_agent(
    llm=llm, tools=tools, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

multi_agent.run("What are 2 famous Taylor Swift songs?")
