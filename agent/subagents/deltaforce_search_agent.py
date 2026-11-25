from agent.prompts import sub_agents_content
from tools.tavily_tool import internet_search
from tools.markdown_tools import generate_markdown
from tools.ragflow_tools import get_assistant_list, create_ask_delete

deltaforce_search_agent = {
    "name":sub_agents_content['deltaforce']['name'],
    "description":sub_agents_content['deltaforce']['description'],
    "system_prompt":sub_agents_content['deltaforce']['system_prompt'],
    "tools":[get_assistant_list, create_ask_delete, internet_search, generate_markdown],
}
