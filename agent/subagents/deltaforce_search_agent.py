from agent.prompts import sub_agents_content
from tools.tavily_tool import internet_search
from tools.markdown_tools import generate_markdown

deltaforce_search_agent = {
    "name":sub_agents_content['deltaforce']['name'],
    "description":sub_agents_content['deltaforce']['description'],
    "system_prompt":sub_agents_content['deltaforce']['system_prompt'],
    "tools":[internet_search, generate_markdown],
}