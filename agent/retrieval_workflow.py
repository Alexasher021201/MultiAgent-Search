"""必须执行的 RAG 前置流程；不让规划模型决定是否跳过。"""

import asyncio
from agent.run_context import RunContext


def select_assistant(ctx: RunContext) -> tuple[str | None, str | None]:
    """配置名称优先，否则只在唯一的匹配候选中选择，避免猜测助手名称。"""
    candidates = ctx.assistants
    if ctx.policy.rag_chat_name:
        name = ctx.policy.rag_chat_name
        return (name, None) if name in candidates else (None, "CONFIGURED_ASSISTANT_NOT_FOUND")
    if ctx.policy.domain == "deltaforce":
        candidates = {
            name: info for name, info in candidates.items()
            if any(term in (name + " " + " ".join(info["datasets"])).lower()
                   for term in ("三角洲", "deltaforce", "delta force"))
        }
    if len(candidates) == 1:
        return next(iter(candidates)), None
    return None, "RAG_ASSISTANT_AMBIGUOUS" if candidates else "NO_RELEVANT_ASSISTANT"


async def run_required_retrieval(ctx: RunContext, discover=None, query=None) -> dict | None:
    """只由服务端调用；两类工具仍经过相同策略、预算与状态记录。"""
    if not ctx.policy.needs_rag:
        return None
    if discover is None or query is None:
        from tools.ragflow_tools import get_assistant_list, create_ask_delete
        discover = discover or get_assistant_list.invoke
        query = query or create_ask_delete.invoke
    ctx.emit("rag_preflight", "执行层开始本轮必需的知识库检索")
    discovered = await asyncio.to_thread(discover, {})
    if discovered["status"] == "error":
        ctx.finish_rag(discovered)
        return discovered
    name, error = select_assistant(ctx)
    if name is None:
        result = {"status": "no_assistant" if error == "NO_RELEVANT_ASSISTANT" else "error",
                  "answer": "", "sources": [], "error_code": error,
                  "message": "未找到唯一的相关助手，请在 execution_policy 中配置真实助手名称"}
        ctx.finish_rag(result)
        return result
    return await asyncio.to_thread(query, {"chat_name": name, "question": ctx.question})
