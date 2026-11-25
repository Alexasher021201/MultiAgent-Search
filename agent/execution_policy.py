"""每轮执行策略。模型只提出工具请求，不能修改这些服务端规则。"""

from dataclasses import dataclass
from enum import StrEnum
import re
from typing import Mapping, Any


class RetrievalMode(StrEnum):
    AUTO = "AUTO"
    DIRECT = "DIRECT"
    RAG_ONLY = "RAG_ONLY"
    RAG_FIRST = "RAG_FIRST"
    WEB_ONLY = "WEB_ONLY"


@dataclass(frozen=True)
class RunPolicy:
    mode: RetrievalMode
    allow_web: bool = True
    max_web_calls: int = 3
    max_rag_calls: int = 3
    max_web_http_attempts: int = 4
    max_tool_calls: int = 20
    max_model_calls: int = 12
    max_denials: int = 3
    timeout_seconds: float = 180
    rag_connect_timeout_seconds: float = 5
    rag_request_timeout_seconds: float = 20
    rag_completion_timeout_seconds: float = 90
    rag_cleanup_timeout_seconds: float = 5
    allow_web_on_rag_error: bool = False
    require_fresh: bool = False
    rag_chat_name: str = ""
    domain: str = "general"

    def __post_init__(self):
        if self.mode == RetrievalMode.AUTO:
            raise ValueError("AUTO 必须在创建运行上下文前解析")
        for name in ("max_web_calls", "max_rag_calls", "max_web_http_attempts"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} 不能为负数")
        for name in ("max_tool_calls", "max_model_calls", "max_denials", "timeout_seconds",
                     "rag_connect_timeout_seconds", "rag_request_timeout_seconds",
                     "rag_completion_timeout_seconds", "rag_cleanup_timeout_seconds"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须大于零")

    @property
    def needs_rag(self) -> bool:
        return self.mode in (RetrievalMode.RAG_ONLY, RetrievalMode.RAG_FIRST)


def resolve_policy(
    question: str,
    requested_mode: str = "AUTO",
    allow_web: bool = True,
    settings: Mapping[str, Any] | None = None,
) -> RunPolicy:
    """保守的首版路由。显式模式优先；禁止联网的文字限制始终收紧权限。

    AUTO 是可解释的关键词规则，不是完整语义分类器；歧义问题可显式选模式。
    次数、超时和故障降级开关仅来自服务端配置，不接受模型参数。
    """
    settings = settings or {}
    mode = RetrievalMode(requested_mode)
    lowered = question.lower()
    delta = any(word in lowered for word in ("三角洲", "delta force", "deltaforce", "威龙"))
    internal = any(word in lowered for word in ("知识库", "内部制度", "sop", "公司政策", "复盘"))
    rag_only = bool(re.search(r"(?:只|仅)(?:能|限)?(?:查|查询|用|使用|从)?.{0,8}(?:知识库|rag)", lowered))
    no_web = bool(re.search(r"(?:不要|禁止|不许|不能|不允许|无需|不用|不使用|别).{0,4}(?:联网|上网|互联网|网络搜索|搜索网络)", lowered))
    basic = bool(re.search(r"(?:知道|了解|听说过).{0,20}(?:吗|么)[？?]?$", question.strip()))
    if rag_only:
        mode = RetrievalMode.RAG_ONLY
    elif mode == RetrievalMode.AUTO:
        if delta and not basic:
            mode = RetrievalMode.RAG_FIRST
        elif internal:
            mode = RetrievalMode.RAG_ONLY
        elif any(word in lowered for word in ("搜索", "搜一下", "查一下", "最新", "新闻", "竞品", "行业趋势","明天","未来某一天")):
            mode = RetrievalMode.WEB_ONLY
        else:
            mode = RetrievalMode.DIRECT
    web_enabled = (bool(settings.get("enable_web", True)) and allow_web and not no_web and not rag_only
                   and mode not in (RetrievalMode.DIRECT, RetrievalMode.RAG_ONLY))
    fields = {
        key: settings[key]
        for key in (
            "max_web_calls", "max_rag_calls", "max_web_http_attempts", "max_tool_calls",
            "max_model_calls", "max_denials", "timeout_seconds", "allow_web_on_rag_error",
            "rag_connect_timeout_seconds", "rag_request_timeout_seconds",
            "rag_completion_timeout_seconds", "rag_cleanup_timeout_seconds",
        )
        if key in settings
    }
    return RunPolicy(
        mode=mode,
        allow_web=web_enabled,
        require_fresh=any(word in lowered for word in ("最新", "今天", "本周", "当前版本", "本赛季")),
        rag_chat_name=str(settings.get("deltaforce_chat_name" if delta else "default_rag_chat_name", "")),
        domain="deltaforce" if delta else "general",
        **fields,
    )
