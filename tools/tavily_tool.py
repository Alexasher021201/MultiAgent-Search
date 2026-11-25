"""网络搜索适配：工具额度和实际 HTTP 重试额度均由本轮执行上下文控制。"""

import os
import time
from typing import Literal

from dotenv import load_dotenv
from langchain_core.tools import tool
from tavily import TavilyClient
from tavily.errors import UsageLimitExceededError, TimeoutError as TavilyTimeoutError
import requests

from agent.run_context import authorize_tool, get_run_context
from api.monitor import monitor

load_dotenv()
tavily_client = None


def _client():
    global tavily_client
    if tavily_client is None:
        tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return tavily_client


def _search_with_retry(query, topic, max_results, include_raw_content):
    ctx = get_run_context()
    for attempt in range(2):
        ctx.reserve_http()
        try:
            return _client().search(query=query, topic=topic, max_results=max_results,
                                    include_raw_content=include_raw_content, timeout=20)
        except (requests.Timeout, requests.ConnectionError, TavilyTimeoutError, UsageLimitExceededError):
            if attempt == 1:
                raise
            time.sleep(0.5)


@tool
def internet_search(
    query: str,
    topic: Literal["news", "finance", "general"] = "general",
    max_results: int = 5,
    include_raw_content: bool = False,
) -> dict:
    """按执行层检索模式搜索公开信息。禁止联网、RAG 未完成或额度不足时不会访问网络。"""
    args = {"query": query, "topic": topic, "max_results": max_results,
            "include_raw_content": include_raw_content}
    with authorize_tool("internet_search", args):
        monitor.report_tool("网络搜索工具", args)
        return _search_with_retry(query, topic, max_results, include_raw_content)
