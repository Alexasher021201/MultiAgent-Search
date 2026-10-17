"""RAGFlow 适配：返回结构化事实，不通过答案关键词猜测是否命中。"""

import logging
from time import monotonic

from langchain_core.tools import tool
from ragflow_sdk import RAGFlow
import requests

from agent.run_context import PolicyViolation, authorize_tool, get_run_context
from api.monitor import monitor
from rawflow.rag_config import _load_ragflow_env

logger = logging.getLogger(__name__)
ragflow_client = None  # 延迟初始化；离线测试可替换为 fake client。

_PHASE_NAMES = {"discovery": "查询助手列表", "resolve_assistant": "确认助手",
                "create_session": "创建临时会话", "ask": "生成知识库回答", "cleanup": "清理临时会话"}


class RagTransportError(Exception):
    """保留安全诊断信息，不把带地址或凭证的上游异常正文发给前端。"""

    def __init__(self, phase, error_type, timed_out, timeout_seconds, elapsed_ms):
        super().__init__(f"RAG HTTP {phase}: {error_type}")
        self.phase = phase
        self.error_type = error_type
        self.timed_out = timed_out
        self.timeout_seconds = timeout_seconds
        self.elapsed_ms = elapsed_ms


class _BoundedRAGFlow(RAGFlow):
    """当前 SDK 没有传输超时；为所有 HTTP 方法设置边界。"""

    def _request(self, method, path, **kwargs):
        ctx = get_run_context()
        phase = ("ask" if path.endswith("/completions") else
                 "cleanup" if method == "DELETE" else
                 "create_session" if method == "POST" and path.endswith("/sessions") else
                 "resolve_assistant" if (kwargs.get("params") or {}).get("id") else "discovery")
        policy = ctx.policy
        remaining = ctx.remaining_seconds()
        read_limit = (policy.rag_completion_timeout_seconds if phase == "ask" else
                      policy.rag_cleanup_timeout_seconds if phase == "cleanup" else
                      policy.rag_request_timeout_seconds)
        connect = min(policy.rag_connect_timeout_seconds, remaining)
        read = min(read_limit, remaining)
        started = monotonic()
        try:
            response = requests.request(method, self.api_url + path, headers=self.authorization_header,
                                        timeout=(connect, read), **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            elapsed_ms = round((monotonic() - started) * 1000)
            timed_out = isinstance(error, requests.Timeout)
            limit = connect if isinstance(error, requests.ConnectTimeout) else read
            logger.warning("RAG HTTP failed run_id=%s phase=%s error_type=%s elapsed_ms=%s timeout_s=%s",
                           ctx.run_id, phase, type(error).__name__, elapsed_ms, limit)
            raise RagTransportError(phase, type(error).__name__, timed_out, limit, elapsed_ms) from error
        finally:
            ctx.emit("rag_http_finished", f"知识库阶段：{_PHASE_NAMES[phase]}，耗时 {monotonic() - started:.2f} 秒",
                     phase=phase, elapsed_ms=round((monotonic() - started) * 1000),
                     connect_timeout_seconds=connect, read_timeout_seconds=read)

    def get(self, path, params=None, json=None):
        return self._request("GET", path, params=params, json=json)

    def post(self, path, json=None, stream=False, files=None):
        return self._request("POST", path, json=json, stream=stream, files=files)

    def delete(self, path, json):
        return self._request("DELETE", path, json=json)

    def put(self, path, json):
        return self._request("PUT", path, json=json)


def _client():
    global ragflow_client
    if ragflow_client is None:
        api_key, base_url = _load_ragflow_env()
        if not api_key or not base_url:
            raise ValueError("缺少 RAGFLOW_API_KEY 或 RAGFLOW_API_URL")
        ragflow_client = _BoundedRAGFlow(api_key=api_key, base_url=base_url)
    return ragflow_client


def _error(error, code="RAG_REQUEST_FAILED", *, phase="ask", elapsed_ms=None):
    # 不向模型或前端暴露上游异常原文。
    timed_out = isinstance(error, (requests.Timeout, TimeoutError)) or getattr(error, "timed_out", False)
    phase = getattr(error, "phase", phase)
    message = (f"知识库在{_PHASE_NAMES.get(phase, phase)}阶段超时，本轮未获得有效检索结果。"
               "请稍后重试，或检查 RAGFlow 模型响应时间及超时配置。") if timed_out else "知识库调用失败，请检查服务配置或运行日志"
    return {"status": "error", "answer": "", "sources": [],
            "error_code": "RAG_TIMEOUT" if timed_out else code,
            "error_type": getattr(error, "error_type", type(error).__name__), "message": message,
            "phase": phase, "elapsed_ms": getattr(error, "elapsed_ms", elapsed_ms),
            "timeout_seconds": getattr(error, "timeout_seconds", None)}


@tool
def get_assistant_list() -> dict:
    """查询真实 RAGFlow 助手及关联知识库。名称必须来自本轮列表；只允许发现一次。"""
    with authorize_tool("get_assistant_list", {}) as ctx:
        monitor.report_tool("ragflow聊天助手列表查询工具：get_assistant_list")
        try:
            chats = _client().list_chats(page_size=100)
            assistants = []
            registry = {}
            duplicate_names = set()
            for chat in chats:
                datasets = [item.get("name", "") if isinstance(item, dict) else getattr(item, "name", "")
                            for item in (getattr(chat, "datasets", None) or [])]
                info = {"id": chat.id, "name": chat.name,
                        "description": getattr(chat, "description", ""), "datasets": datasets}
                assistants.append(info)
                if chat.name in registry:
                    duplicate_names.add(chat.name)
                registry[chat.name] = info
            for name in duplicate_names:
                registry.pop(name, None)
            if len(chats) >= 100:
                return _error(ValueError("列表可能被截断"), "RAG_DISCOVERY_TRUNCATED")
            if duplicate_names:
                return _error(ValueError("助手名称重复"), "RAG_DUPLICATE_NAMES")
            ctx.set_assistants(registry)
            return {"status": "success" if assistants else "no_assistant", "assistants": assistants,
                    "ambiguous_names": sorted(duplicate_names), "error_code": None}
        except PolicyViolation:
            raise
        except Exception as error:
            return _error(error, "RAG_DISCOVERY_FAILED", phase="discovery")


@tool
def create_ask_delete(chat_name: str, question: str) -> dict:
    """用本轮真实助手查询原始问题，返回答案、引用和状态；结束后清理临时会话。"""
    # 原始问题已经由固定工作流传入，不再用正则擅自删改用户问题。
    question = question.strip()
    with authorize_tool("create_ask_delete", {"chat_name": chat_name, "question": question}) as ctx:
        monitor.report_tool("ragflow提问助手工具：create_ask_delete", {"chat_name": chat_name})
        chat = session = None
        phase = "resolve_assistant"
        started = monotonic()
        try:
            info = ctx.assistants[chat_name]
            chats = _client().list_chats(id=info["id"])
            matches = [item for item in chats if item.id == info["id"]]
            if len(matches) != 1:
                raise ValueError("已发现的助手不存在或标识不唯一")
            chat = matches[0]
            phase = "create_session"
            session = chat.create_session(name=f"run_{ctx.run_id}")
            # SDK 的非流式 ask 也返回生成器，避免无限续流绕过读超时。
            answer = ""
            references = None
            phase = "ask"
            for part in session.ask(question=question, stream=False):
                answer = getattr(part, "content", "") or ""
                if getattr(part, "reference", None) is not None:
                    references = part.reference
            sources = []
            for ref in references or []:
                if not isinstance(ref, dict):
                    continue
                text = ref.get("content") or ref.get("content_with_weight") or ""
                if text:
                    sources.append({"document_id": ref.get("document_id") or ref.get("doc_id"),
                                    "document_name": ref.get("document_name") or ref.get("docnm_kwd"),
                                    "content": str(text)[:4000]})
            status = "success" if answer.strip() and sources else (
                "no_hit" if references == [] or not answer.strip() else "unverified")
            result = {"status": status, "answer": answer[:16000], "sources": sources[:8],
                      "assistant": chat_name, "question": question, "error_code": None}
        except PolicyViolation:
            raise
        except Exception as error:
            result = _error(error, phase=phase, elapsed_ms=round((monotonic() - started) * 1000))
        finally:
            if chat is not None and session is not None:
                try:
                    chat.delete_sessions(ids=[session.id])
                except Exception as error:
                    logger.warning("RAG 临时会话清理失败 run_id=%s error_type=%s", ctx.run_id, type(error).__name__)
        ctx.finish_rag(result)
        detail = f"（{result.get('error_type')}，{_PHASE_NAMES.get(result.get('phase'), '')}阶段）" if result["status"] == "error" else ""
        ctx.emit("rag_outcome", f"知识库查询状态：{result['status']}{detail}", status=result["status"],
                 error_code=result.get("error_code"), phase=result.get("phase"),
                 error_type=result.get("error_type"), elapsed_ms=result.get("elapsed_ms"),
                 timeout_seconds=result.get("timeout_seconds"))
        return result
