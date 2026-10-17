"""离线回归：使用真实 LangChain/Deep Agents，所有外部客户端都替换为 fake。"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from deepagents import create_deep_agent
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage

from agent.execution_policy import RetrievalMode, RunPolicy, resolve_policy
from agent.policy_middleware import ToolPolicyMiddleware
from agent.retrieval_workflow import run_required_retrieval
from agent.run_context import PolicyViolation, RunContext, RunStopped, authorize_tool, use_run_context
from tools import ragflow_tools as rag
from tools import tavily_tool as web


class ToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class FakeSession:
    id = "fake-session"

    def __init__(self, answer="有依据的回答", references=None, error=None):
        self.answer = answer
        self.references = references
        self.error = error
        self.questions = []

    def ask(self, question, stream=False):
        self.questions.append(question)
        if self.error:
            raise self.error
        yield SimpleNamespace(content=self.answer, reference=self.references)


class FakeChat:
    id = "fake-chat"
    name = "三角洲资料助手"
    description = "游戏资料"
    datasets = [{"name": "三角洲知识库"}]

    def __init__(self, session):
        self.session = session
        self.deleted = []

    def create_session(self, name):
        return self.session

    def delete_sessions(self, ids):
        self.deleted.extend(ids)


def make_client(references=None, error=None):
    session = FakeSession(references=references, error=error)
    chat = FakeChat(session)
    client = Mock()
    client.list_chats.return_value = [chat]
    return client, chat


def context(mode="RAG_FIRST", **kwargs):
    return RunContext(RunPolicy(RetrievalMode(mode), **kwargs), "三角洲威龙擅长什么枪械")


def call(name, args, ident="call-1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": ident, "type": "tool_call"}])


class PolicyTests(unittest.TestCase):
    def test_explicit_restrictions_and_server_switch_override_mode(self):
        self.assertEqual(resolve_policy("只查知识库：威龙", "WEB_ONLY").mode, RetrievalMode.RAG_ONLY)
        self.assertFalse(resolve_policy("禁止联网，查最新资讯", "WEB_ONLY").allow_web)
        self.assertFalse(resolve_policy("搜索新闻", settings={"enable_web": False}).allow_web)
        self.assertFalse(resolve_policy("搜索新闻", allow_web=False).allow_web)

    def test_auto_is_conservative(self):
        for question, mode in [("你好", "DIRECT"), ("你知道三角洲吗", "DIRECT"),
                               ("三角洲威龙擅长什么枪械", "RAG_FIRST"),
                               ("查询内部制度知识库", "RAG_ONLY"), ("搜索今天新闻", "WEB_ONLY")]:
            self.assertEqual(resolve_policy(question).mode.value, mode)

    def test_missing_context_fails_closed(self):
        with patch.object(web, "_client") as client:
            with self.assertRaises(PolicyViolation):
                web.internet_search.invoke({"query": "test"})
            client.assert_not_called()

    def test_raw_tool_cannot_bypass_rag_or_no_web(self):
        for ctx in (context(), context("RAG_ONLY"), context("DIRECT"), context("WEB_ONLY", allow_web=False)):
            with self.subTest(mode=ctx.policy.mode), use_run_context(ctx), patch.object(web, "_client") as client:
                with self.assertRaises(PolicyViolation):
                    web.internet_search.invoke({"query": "test"})
                client.assert_not_called()

    def test_only_discovery_does_not_unlock_web_and_question_is_scoped(self):
        ctx = context()
        ctx.set_assistants({"fake": {"id": "id"}})
        with use_run_context(ctx):
            with self.assertRaisesRegex(PolicyViolation, "必须先完成"):
                ctx.reserve("internet_search", {"query": "test"})
            with self.assertRaisesRegex(PolicyViolation, "原始问题"):
                ctx.reserve("create_ask_delete", {"chat_name": "fake", "question": "无关问题"})

    def test_atomic_thread_reservation_and_failed_calls_spend_budget(self):
        ctx = context("WEB_ONLY", max_denials=100)
        def worker(index):
            with use_run_context(ctx):
                try:
                    with authorize_tool("internet_search", {"query": str(index)}):
                        if index == 0:
                            raise RuntimeError("fake timeout")
                    return True
                except RuntimeError:
                    return True
                except PolicyViolation:
                    return False
        with ThreadPoolExecutor(max_workers=10) as pool:
            self.assertEqual(sum(pool.map(worker, range(20))), 2)
        self.assertEqual(ctx.counts["web"], 2)

    def test_duplicate_search_and_general_purpose_are_blocked(self):
        ctx = context("WEB_ONLY")
        ctx.reserve("internet_search", {"query": " Delta   Force "})
        with self.assertRaises(PolicyViolation):
            ctx.reserve("internet_search", {"query": "delta force"})
        with self.assertRaises(PolicyViolation):
            ctx.reserve("task", {"subagent_type": "general-purpose"})

    def test_denials_model_and_tool_budgets_stop_execution(self):
        ctx = context("RAG_ONLY", max_denials=2)
        with self.assertRaises(PolicyViolation):
            ctx.reserve("internet_search", {"query": "one"})
        with self.assertRaises(RunStopped):
            ctx.reserve("internet_search", {"query": "two"})
        ctx = context("DIRECT", max_model_calls=1, max_tool_calls=1)
        ctx.reserve_model()
        with self.assertRaises(RunStopped):
            ctx.reserve_model()
        ctx.reserve("read_file_content", {})
        with self.assertRaises(RunStopped):
            ctx.reserve("read_file_content", {})

    def test_closed_and_expired_runs_reject_late_work(self):
        ctx = context()
        ctx.close()
        with self.assertRaises(RunStopped):
            ctx.finish_rag({"status": "no_hit"})
        ctx = context("WEB_ONLY", timeout_seconds=1)
        ctx._started -= 2
        with self.assertRaises(RunStopped):
            ctx.reserve("internet_search", {"query": "late"})


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_formatted_user_question_is_not_silently_rewritten(self):
        client, chat = make_client([])
        question = '查询并回答：“威龙擅长什么枪械？” 要求：只查知识库'
        ctx = RunContext(RunPolicy(RetrievalMode.RAG_ONLY), question)
        with use_run_context(ctx), patch.object(rag, "ragflow_client", client):
            await run_required_retrieval(ctx)
        self.assertEqual(chat.session.questions, [question])

    async def test_cited_result_blocks_unnecessary_web_and_session_is_cleaned(self):
        client, chat = make_client([{"document_name": "角色说明", "content": "威龙资料"}])
        ctx = context()
        with use_run_context(ctx), patch.object(rag, "ragflow_client", client), patch.object(web, "_client") as search:
            result = await run_required_retrieval(ctx)
            self.assertEqual(result["status"], "success")
            self.assertEqual(chat.session.questions, [ctx.question])
            self.assertEqual(chat.deleted, [chat.session.id])
            with self.assertRaises(PolicyViolation):
                web.internet_search.invoke({"query": "unnecessary"})
            search.assert_not_called()

    async def test_no_hit_unverified_and_error_remain_distinct(self):
        for references, error, expected in [([], None, "no_hit"), (None, None, "unverified"),
                                             (None, TimeoutError("secret must not leak"), "error")]:
            client, chat = make_client(references, error)
            ctx = context()
            with self.subTest(status=expected), use_run_context(ctx), patch.object(rag, "ragflow_client", client):
                result = await run_required_retrieval(ctx)
                self.assertEqual(result["status"], expected)
                self.assertNotIn("secret", json.dumps(result))
                self.assertEqual(chat.deleted, [chat.session.id])
                if expected == "error":
                    with self.assertRaises(PolicyViolation):
                        ctx.reserve("internet_search", {"query": "fallback"})
                else:
                    ctx.reserve("internet_search", {"query": "fallback"})

    async def test_rag_only_never_unlocks_web_after_error_or_no_hit(self):
        for references, error in [([], None), (None, TimeoutError())]:
            client, _ = make_client(references, error)
            ctx = context("RAG_ONLY", allow_web_on_rag_error=True)
            with use_run_context(ctx), patch.object(rag, "ragflow_client", client):
                await run_required_retrieval(ctx)
                with self.assertRaises(PolicyViolation):
                    web.internet_search.invoke({"query": "fallback"})

    async def test_error_fallback_requires_server_setting(self):
        ctx = context(allow_web_on_rag_error=True)
        ctx.finish_rag({"status": "error"})
        ctx.reserve("internet_search", {"query": "fallback"})

    async def test_discovery_error_and_ambiguous_configuration_do_not_masquerade_as_no_hit(self):
        client = Mock()
        client.list_chats.side_effect = TimeoutError()
        ctx = context()
        with use_run_context(ctx), patch.object(rag, "ragflow_client", client):
            result = await run_required_retrieval(ctx)
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["error_code"], "RAG_TIMEOUT")
            self.assertEqual(result["phase"], "discovery")
        client, chat = make_client([])
        client.list_chats.return_value = [chat, SimpleNamespace(id="other", name="三角洲第二助手", datasets=[])]
        ctx = context()
        with use_run_context(ctx), patch.object(rag, "ragflow_client", client):
            result = await run_required_retrieval(ctx)
            self.assertEqual(result["error_code"], "RAG_ASSISTANT_AMBIGUOUS")
            self.assertEqual(result["status"], "error")

    async def test_no_assistant_is_explicit_and_freshness_can_allow_supplement(self):
        client = Mock()
        client.list_chats.return_value = []
        ctx = context()
        with use_run_context(ctx), patch.object(rag, "ragflow_client", client):
            result = await run_required_retrieval(ctx)
            self.assertEqual(result["status"], "no_assistant")
            ctx.reserve("internet_search", {"query": "fallback"})
        ctx = context(require_fresh=True)
        ctx.finish_rag({"status": "success"})
        ctx.reserve("internet_search", {"query": "latest"})

    async def test_per_run_context_and_worker_propagation_are_isolated(self):
        async def work(question):
            ctx = RunContext(RunPolicy(RetrievalMode.WEB_ONLY), question)
            with use_run_context(ctx):
                def reserve():
                    with authorize_tool("internet_search", {"query": question}):
                        return ctx.summary()
                await asyncio.sleep(0)
                return await asyncio.to_thread(reserve)
        first, second = await asyncio.gather(work("first"), work("second"))
        self.assertNotEqual(first["run_id"], second["run_id"])
        self.assertEqual(first["counts"]["web"], 1)
        self.assertEqual(second["counts"]["web"], 1)


class FrameworkTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_async_toolnode_blocks_fake_model_web_request(self):
        model = ToolModel(responses=[call("internet_search", {"query": "blocked"}), AIMessage(content="资料不足")])
        agent = create_agent(model, tools=[web.internet_search], middleware=[ToolPolicyMiddleware()])
        ctx = context("RAG_ONLY")
        with use_run_context(ctx), patch.object(web, "_client") as search:
            result = await agent.ainvoke({"messages": [{"role": "user", "content": ctx.question}]})
        search.assert_not_called()
        errors = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual(json.loads(errors[0].content)["error_code"], "WEB_FORBIDDEN")

    async def test_real_toolnode_claims_one_budget_with_default_arguments(self):
        model = ToolModel(responses=[call("internet_search", {"query": "allowed"}), AIMessage(content="完成")])
        agent = create_agent(model, tools=[web.internet_search], middleware=[ToolPolicyMiddleware()])
        ctx = context("WEB_ONLY")
        client = Mock()
        client.search.return_value = {"results": []}
        with use_run_context(ctx), patch.object(web, "tavily_client", client):
            await agent.ainvoke({"messages": [{"role": "user", "content": "search"}]})
        self.assertEqual(ctx.counts["web"], 1)
        self.assertEqual(ctx.counts["web_http"], 1)
        client.search.assert_called_once()

    async def test_parallel_model_calls_cannot_exceed_shared_budget(self):
        message = AIMessage(content="", tool_calls=[{"name": "internet_search", "args": {"query": str(i)},
                            "id": f"call-{i}", "type": "tool_call"} for i in range(3)])
        model = ToolModel(responses=[message, AIMessage(content="完成")])
        agent = create_agent(model, tools=[web.internet_search], middleware=[ToolPolicyMiddleware()])
        ctx = context("WEB_ONLY")
        client = Mock()
        client.search.return_value = {"results": []}
        with use_run_context(ctx), patch.object(web, "tavily_client", client):
            await agent.ainvoke({"messages": [{"role": "user", "content": "search"}]})
        self.assertEqual(client.search.call_count, 2)
        self.assertEqual(ctx.counts["web"], 2)

    async def test_switching_subagent_does_not_bypass_rag_only(self):
        child = ToolModel(responses=[call("internet_search", {"query": "bypass"}), AIMessage(content="已受限")])
        root = ToolModel(responses=[call("task", {"subagent_type": "searcher", "description": "search"}),
                                    AIMessage(content="完成")])
        agent = create_deep_agent(root, middleware=[ToolPolicyMiddleware()], subagents=[{
            "name": "searcher", "description": "search", "system_prompt": "search",
            "model": child, "tools": [web.internet_search], "middleware": [ToolPolicyMiddleware()],
        }])
        ctx = context("RAG_ONLY")
        with use_run_context(ctx), patch.object(web, "_client") as search:
            await agent.ainvoke({"messages": [{"role": "user", "content": "question"}]})
        search.assert_not_called()
        self.assertEqual(ctx.counts["models"], 4)

    async def test_sync_toolnode_is_also_protected(self):
        model = ToolModel(responses=[call("internet_search", {"query": "blocked"}), AIMessage(content="完成")])
        agent = create_agent(model, tools=[web.internet_search], middleware=[ToolPolicyMiddleware()])
        with use_run_context(context("DIRECT")), patch.object(web, "_client") as search:
            result = await asyncio.to_thread(agent.invoke, {"messages": [{"role": "user", "content": "hello"}]})
        search.assert_not_called()
        self.assertEqual(result["messages"][-1].content, "完成")

    async def test_repeated_denials_really_terminate_framework_loop(self):
        model = ToolModel(responses=[call("internet_search", {"query": "blocked"}, f"call-{i}") for i in range(3)])
        agent = create_agent(model, tools=[web.internet_search], middleware=[ToolPolicyMiddleware()])
        with use_run_context(context("DIRECT", max_denials=2)):
            with self.assertRaises(RunStopped):
                await agent.ainvoke({"messages": [{"role": "user", "content": "hello"}]})


class HttpBudgetTests(unittest.TestCase):
    def test_http_budget_caps_total_retries(self):
        from tavily.errors import TimeoutError as TavilyTimeoutError
        ctx = context("WEB_ONLY", max_web_http_attempts=1)
        client = Mock()
        client.search.side_effect = TavilyTimeoutError(20)
        with use_run_context(ctx), patch.object(web, "tavily_client", client), patch.object(web.time, "sleep"):
            with self.assertRaisesRegex(PolicyViolation, "HTTP"):
                web.internet_search.invoke({"query": "test"})
        self.assertEqual(client.search.call_count, 1)

    def test_retry_consumes_http_budget_and_auth_errors_are_not_retried(self):
        from tavily.errors import InvalidAPIKeyError, TimeoutError as TavilyTimeoutError
        for error, expected in [(TavilyTimeoutError(20), 2), (InvalidAPIKeyError("invalid"), 1)]:
            ctx = context("WEB_ONLY")
            client = Mock()
            client.search.side_effect = error
            with use_run_context(ctx), patch.object(web, "tavily_client", client), patch.object(web.time, "sleep"):
                with self.assertRaises(type(error)):
                    web.internet_search.invoke({"query": "test"})
            self.assertEqual(client.search.call_count, expected)
            self.assertEqual(ctx.counts["web"], 1)
            self.assertEqual(ctx.counts["web_http"], expected)


class RunLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_main_run_performs_rag_before_model_can_answer_directly(self):
        import agent.main_agent as runner
        model = ToolModel(responses=[AIMessage(content="已经读取资料")])
        client, _ = make_client([{"document_name": "角色文档", "content": "真实知识库片段"}])
        events = []
        with tempfile.TemporaryDirectory() as temporary, patch.object(runner, "project_root_path", Path(temporary)), \
             patch.object(runner, "model", model), patch.object(rag, "ragflow_client", client), \
             patch.object(runner, "prompt_yaml_content", {"execution_policy": {}}), \
             patch.object(runner.monitor, "_emit", side_effect=lambda event, message, data=None: events.append((event, data))):
            agent = runner.create_main_agent(None)
            await runner.run_deep_agent("三角洲威龙擅长什么枪械", "test-session", agent, "RAG_ONLY")
        names = [event for event, _ in events]
        self.assertNotIn("error", names)
        self.assertLess(names.index("rag_outcome"), names.index("model_start"))
        self.assertLess(names.index("model_start"), names.index("task_result"))
        summary = next(data for event, data in events if event == "run_summary")
        self.assertEqual(summary["counts"]["rag"], 1)
        self.assertEqual(summary["counts"]["models"], 1)
        self.assertEqual(summary["counts"].get("web", 0), 0)

    async def test_total_timeout_reports_failure_and_closes_run(self):
        import agent.main_agent as runner
        class SlowGraph:
            async def astream(self, *args, **kwargs):
                await asyncio.sleep(0.2)
                yield {"model": {"messages": [AIMessage(content="too late")]}}
        events = []
        with tempfile.TemporaryDirectory() as temporary, patch.object(runner, "project_root_path", Path(temporary)), \
             patch.object(runner, "prompt_yaml_content", {"execution_policy": {"timeout_seconds": 0.02}}), \
             patch.object(runner.monitor, "_emit", side_effect=lambda event, message, data=None: events.append((event, data))):
            await runner.run_deep_agent("hello", "test-session", SlowGraph(), "DIRECT")
        names = [event for event, _ in events]
        self.assertIn("error", names)
        self.assertNotIn("task_result", names)
        self.assertIn("run_summary", names)
        from agent.run_context import get_run_context
        with self.assertRaises(PolicyViolation):
            get_run_context()

    async def test_api_validates_mode_and_passes_restriction_to_runner(self):
        import api.server as server
        import httpx
        from unittest.mock import AsyncMock
        fake = AsyncMock()
        server.app.state.main_agent = object()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test") as client:
            with patch.object(server, "run_deep_agent", fake):
                bad = await client.post("/api/task", json={"query": "test", "retrieval_mode": "WRONG"})
                self.assertEqual(bad.status_code, 422)
                response = await client.post("/api/task", json={"query": "test", "retrieval_mode": "RAG_ONLY", "allow_web": False})
                await asyncio.sleep(0)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(fake.await_args.args[3:5], ("RAG_ONLY", False))
        self.assertEqual(fake.await_args.args[5], response.json()["run_id"])


if __name__ == "__main__":
    unittest.main()
