import yaml
from langchain.agents.agent_toolkits.openapi.spec import reduce_openapi_spec
import spotipy.util as util
from langchain.requests import RequestsWrapper
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits.openapi import planner

with open("spotify_openapi.yaml") as f:
    raw_spotify_api_spec = yaml.load(f, Loader=yaml.Loader)
spotify_api_spec = reduce_openapi_spec(raw_spotify_api_spec)

def construct_spotify_auth_headers(raw_spec: dict):
    scopes = list(raw_spec['components']['securitySchemes']['oauth_2_0']['flows']['authorizationCode']['scopes'].keys())
    access_token = util.prompt_for_user_token(scope=','.join(scopes))
    # print(access_token)
    return {
        'Authorization': f'Bearer {access_token}'
    }

# Get API credentials.
headers = construct_spotify_auth_headers(raw_spotify_api_spec)
requests_wrapper = RequestsWrapper(headers=headers)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

spotify_agent = planner.create_openapi_agent(spotify_api_spec, requests_wrapper, llm)
user_query = "Can you show taylor swift songs?"
spotify_agent.run(user_query)