"""协程传递、线程安全的本轮状态；所有助手共享同一个上下文对象。"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
from threading import Lock
from time import monotonic
from typing import Any, Callable
import uuid

from agent.execution_policy import RetrievalMode, RunPolicy


class PolicyViolation(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code

    def payload(self):
        return {"status": "blocked", "error_code": self.code, "message": str(self), "retryable": False}


class RunStopped(PolicyViolation):
    """终止本轮，而不是把错误交给模型继续反复尝试。"""


@dataclass
class RunContext:
    policy: RunPolicy
    question: str
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_sink: Callable[[str, str, dict], None] | None = field(default=None, repr=False)
    counts: dict[str, int] = field(default_factory=dict)
    assistants: dict[str, dict] = field(default_factory=dict)
    discovery_done: bool = False
    rag_stage: str = "not_started"
    rag_outcome: dict | None = None
    closed: bool = False
    _started: float = field(default_factory=monotonic, repr=False)
    _lock: Any = field(default_factory=Lock, repr=False)
    _queries: set[str] = field(default_factory=set, repr=False)
    _events: list[dict] = field(default_factory=list, repr=False)

    def emit(self, event: str, message: str, **data):
        payload = {"run_id": self.run_id, **data}
        with self._lock:
            self._events.append({"event": event, **payload})
        if self.event_sink:
            self.event_sink(event, message, payload)

    def _live(self):
        if self.closed:
            raise RunStopped("RUN_CLOSED", "本轮已经结束")
        if monotonic() - self._started >= self.policy.timeout_seconds:
            raise RunStopped("RUN_TIMEOUT", "本轮执行时间已耗尽")

    def _deny(self, code: str, message: str):
        self.counts["denials"] = self.counts.get("denials", 0) + 1
        if self.counts["denials"] >= self.policy.max_denials:
            raise RunStopped("TOO_MANY_DENIALS", f"本轮违规请求达到上限：{message}")
        raise PolicyViolation(code, message)

    def _web_allowed(self):
        p = self.policy
        if not p.allow_web or p.mode in (RetrievalMode.DIRECT, RetrievalMode.RAG_ONLY):
            self._deny("WEB_FORBIDDEN", "本轮禁止联网，请根据已有资料回答或说明限制")
        if p.needs_rag:
            if self.rag_stage != "completed" or self.rag_outcome is None:
                self._deny("RAG_REQUIRED", "必须先完成本轮原始问题的知识库检索")
            status = self.rag_outcome["status"]
            if status == "error" and not p.allow_web_on_rag_error:
                self._deny("RAG_ERROR_NO_FALLBACK", "知识库调用失败，本轮未启用故障联网降级")
            if status == "success" and not p.require_fresh:
                self._deny("RAG_EVIDENCE_AVAILABLE", "已有带引用的知识库结果，本轮不允许额外搜索")
            if status not in ("success", "no_hit", "unverified", "no_assistant", "error"):
                self._deny("INVALID_RAG_OUTCOME", "知识库状态不合法，不能解锁网络搜索")

    def reserve(self, tool: str, args: dict) -> str:
        """检查与预占在同一锁内，网络 I/O 在锁外。失败调用也消耗额度。"""
        with self._lock:
            self._live()
            if self.counts.get("tools", 0) >= self.policy.max_tool_calls:
                raise RunStopped("TOOL_BUDGET", "本轮工具调用预算已耗尽")
            if tool == "task" and args.get("subagent_type") == "general-purpose":
                self._deny("UNMANAGED_AGENT", "通用助手未接入本项目策略链路，请使用已注册的业务助手")
            if tool == "internet_search":
                self._web_allowed()
                query = " ".join(str(args.get("query", "")).lower().split())
                if not query:
                    self._deny("EMPTY_QUERY", "搜索词不能为空")
                if not 1 <= args.get("max_results", 5) <= 10:
                    self._deny("INVALID_MAX_RESULTS", "max_results 必须在 1 到 10 之间")
                if query in self._queries:
                    self._deny("DUPLICATE_QUERY", "本轮已执行相同搜索，请使用已有结果")
                if self.counts.get("web", 0) >= self.policy.max_web_calls:
                    self._deny("WEB_BUDGET", "本轮搜索工具调用额度已耗尽")
                self.counts["web"] = self.counts.get("web", 0) + 1
                self._queries.add(query)
            elif tool in ("get_assistant_list", "create_ask_delete"):
                if not self.policy.needs_rag:
                    self._deny("RAG_FORBIDDEN", "当前检索模式未启用知识库，请重新选择模式")
                if tool == "get_assistant_list":
                    if self.counts.get("discovery", 0):
                        self._deny("DISCOVERY_ALREADY_DONE", "已查询过助手列表，请使用执行层提供的结果")
                    self.counts["discovery"] = 1
                else:
                    if not self.discovery_done:
                        self._deny("DISCOVERY_REQUIRED", "请先完成真实助手列表查询")
                    if args.get("question", "").strip() != self.question.strip():
                        self._deny("QUESTION_SCOPE", "首版检索只允许本轮原始问题，不能用其他问题解锁联网")
                    if args.get("chat_name") not in self.assistants:
                        self._deny("UNKNOWN_ASSISTANT", "助手名称未通过本轮真实列表验证")
                    if self.rag_stage == "completed":
                        self._deny("RAG_ALREADY_DONE", "本轮已完成检索，请使用执行层提供的结果")
                    if self.rag_stage == "running":
                        self._deny("RAG_IN_PROGRESS", "本轮知识库查询正在执行，不能并发重复查询")
                    if self.counts.get("rag", 0) >= self.policy.max_rag_calls:
                        self._deny("RAG_BUDGET", "本轮知识库查询额度已耗尽")
                    self.counts["rag"] = self.counts.get("rag", 0) + 1
                    self.rag_stage = "running"
            self.counts["tools"] = self.counts.get("tools", 0) + 1
            return str(uuid.uuid4())

    def reserve_model(self):
        with self._lock:
            self._live()
            if self.counts.get("models", 0) >= self.policy.max_model_calls:
                raise RunStopped("MODEL_BUDGET", "本轮模型调用预算已耗尽")
            self.counts["models"] = self.counts.get("models", 0) + 1

    def reserve_http(self):
        with self._lock:
            self._live()
            if self.counts.get("web_http", 0) >= self.policy.max_web_http_attempts:
                raise PolicyViolation("WEB_HTTP_BUDGET", "本轮实际搜索 HTTP 请求额度已耗尽")
            self.counts["web_http"] = self.counts.get("web_http", 0) + 1

    def remaining_seconds(self) -> float:
        with self._lock:
            self._live()
            return max(0.001, self.policy.timeout_seconds - (monotonic() - self._started))

    def set_assistants(self, assistants: dict[str, dict]):
        with self._lock:
            self._live()
            self.assistants = assistants
            self.discovery_done = True

    def finish_rag(self, outcome: dict):
        with self._lock:
            self._live()
            self.rag_outcome = outcome
            self.rag_stage = "completed"

    def close(self):
        with self._lock:
            self.closed = True

    def summary(self):
        with self._lock:
            return {"run_id": self.run_id, "mode": self.policy.mode.value,
                    "allow_web": self.policy.allow_web, "counts": dict(self.counts),
                    "rag_stage": self.rag_stage,
                    "rag_status": (self.rag_outcome or {}).get("status")}


_current: ContextVar[RunContext | None] = ContextVar("execution_run", default=None)


def get_run_context() -> RunContext:
    ctx = _current.get()
    if ctx is None:
        raise PolicyViolation("MISSING_RUN_CONTEXT", "缺少服务端执行上下文，拒绝调用受控工具")
    return ctx


@contextmanager
def use_run_context(ctx: RunContext):
    token = _current.set(ctx)
    try:
        yield ctx
    finally:
        _current.reset(token)


@dataclass
class _Lease:
    run_id: str
    fingerprint: str
    claimed: bool = False
    lock: Any = field(default_factory=Lock)


_lease: ContextVar[_Lease | None] = ContextVar("tool_lease", default=None)


@contextmanager
def authorize_tool(name: str, args: dict, *, middleware: bool = False):
    """中间件预占一次；工具入口认领一次，避免双计数及直接调用绕过。"""
    ctx = get_run_context()
    fingerprint_args = dict(args)
    if name == "internet_search":
        fingerprint_args = {"topic": "general", "max_results": 5, "include_raw_content": False, **args}
    fingerprint = json.dumps([name, fingerprint_args], sort_keys=True, ensure_ascii=False, default=str)
    lease = _lease.get()
    if not middleware and lease and lease.run_id == ctx.run_id and lease.fingerprint == fingerprint:
        with lease.lock:
            reuse = not lease.claimed
            lease.claimed = True
        if reuse:
            yield ctx
            return
    try:
        call_id = ctx.reserve(name, args)
    except PolicyViolation as error:
        ctx.emit("policy_blocked", str(error), tool=name, error_code=error.code)
        raise
    token = _lease.set(_Lease(ctx.run_id, fingerprint, claimed=not middleware))
    ctx.emit("policy_allowed", f"执行层放行工具：{name}", tool=name, call_id=call_id)
    try:
        yield ctx
    except BaseException as error:
        ctx.emit("tool_finished", f"工具调用未完成：{name}", tool=name, call_id=call_id,
                 status="error", error_type=type(error).__name__)
        raise
    else:
        ctx.emit("tool_finished", f"工具调用结束：{name}", tool=name, call_id=call_id, status="returned")
    finally:
        _lease.reset(token)
