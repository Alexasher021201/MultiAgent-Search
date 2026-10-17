"""用本地 HTTP 服务复现非流式第二次慢响应，不访问线上 RAGFlow 或模型。"""

import asyncio
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
from threading import Thread
import time
import unittest
from unittest.mock import patch
import uuid

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent.execution_policy import RetrievalMode, RunPolicy, resolve_policy
from agent.run_context import RunContext, use_run_context
from tools import ragflow_tools as rag


class ToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@contextmanager
def local_rag_server():
    state = {"asks": 0, "session_names": [], "deleted_ids": []}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, data):
            body = json.dumps({"code": 0, "data": data}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # 短超时测试会主动关闭客户端连接。

        def payload(self):
            return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")

        def do_GET(self):
            self.respond([{"id": "chat-1", "name": "三角洲", "description": "游戏资料",
                           "datasets": [{"name": "三角洲知识库"}]}])

        def do_POST(self):
            payload = self.payload()
            if self.path.endswith("/completions"):
                state["asks"] += 1
                if state["asks"] == 2:
                    time.sleep(0.25)
                self.respond({"answer": "威龙资料", "reference": {"chunks": [
                    {"document_name": "威龙.txt", "content": "威龙使用冲锋枪"}]}})
            else:
                state["session_names"].append(payload["name"])
                self.respond({"id": str(uuid.uuid4()), "chat_id": "chat-1", "name": payload["name"]})

        def do_DELETE(self):
            state["deleted_ids"].extend(self.payload()["ids"])
            self.respond(True)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


class RagTransportTests(unittest.TestCase):
    def test_completion_uses_separate_configured_timeout_and_remaining_run_budget(self):
        policy = resolve_policy("威龙", "RAG_ONLY", settings={"rag_completion_timeout_seconds": 75})
        self.assertEqual(policy.rag_completion_timeout_seconds, 75)
        ctx = RunContext(policy, "威龙")
        client = rag._BoundedRAGFlow("fake-key", "http://example.invalid")
        with use_run_context(ctx), patch.object(rag.requests, "request") as request:
            client.post("/chats/id/completions", json={})
            self.assertEqual(request.call_args.kwargs["timeout"], (5, 75))
            client.get("/chats")
            self.assertEqual(request.call_args.kwargs["timeout"], (5, 20))
            client.delete("/chats/id/sessions", {})
            self.assertEqual(request.call_args.kwargs["timeout"], (5, 5))
            ctx._started -= 170
            client.post("/chats/id/completions", json={})
            self.assertLessEqual(request.call_args.kwargs["timeout"][1], 10)

    def test_transport_timeout_exposes_phase_without_leaking_exception_text(self):
        ctx = RunContext(RunPolicy(RetrievalMode.RAG_ONLY), "威龙")
        client = rag._BoundedRAGFlow("fake-key", "http://example.invalid")
        with use_run_context(ctx), patch.object(rag.requests, "request", side_effect=rag.requests.ReadTimeout("secret-token")):
            with self.assertRaises(rag.RagTransportError) as caught:
                client.post("/chats/id/completions", json={})
        outcome = rag._error(caught.exception)
        self.assertEqual(outcome["error_code"], "RAG_TIMEOUT")
        self.assertEqual(outcome["phase"], "ask")
        self.assertEqual(outcome["error_type"], "ReadTimeout")
        self.assertEqual(outcome["timeout_seconds"], 90)
        self.assertNotIn("secret-token", json.dumps(outcome))


class ConsecutiveRagTests(unittest.IsolatedAsyncioTestCase):
    async def run_two_turns(self, completion_timeout):
        import agent.main_agent as runner
        model = ToolModel(responses=[AIMessage(content="第一轮真实结果"), AIMessage(content="第二轮真实结果")])
        settings = {"timeout_seconds": 10, "rag_request_timeout_seconds": 0.1,
                    "rag_completion_timeout_seconds": completion_timeout,
                    "deltaforce_chat_name": "三角洲"}
        events = []
        with local_rag_server() as (url, state), tempfile.TemporaryDirectory() as temporary, \
             patch.object(runner, "project_root_path", Path(temporary)), patch.object(runner, "model", model), \
             patch.object(runner, "prompt_yaml_content", {"execution_policy": settings}), \
             patch.object(rag, "ragflow_client", rag._BoundedRAGFlow("fake-key", url)), \
             patch.object(runner.monitor, "_emit", side_effect=lambda event, message, data=None: events.append((event, message, data))):
            async with AsyncSqliteSaver.from_conn_string(str(Path(temporary) / "checkpoints.db")) as saver:
                await saver.setup()
                agent = runner.create_main_agent(saver)
                await runner.run_deep_agent("威龙擅长什么枪械", "same-session", agent, "RAG_ONLY")
                first_events = list(events)
                events.clear()
                await runner.run_deep_agent("威龙擅长什么枪械", "same-session", agent, "RAG_ONLY")
                second_events = list(events)
        return first_events, second_events, state

    async def test_same_checkpoint_two_turns_succeed_when_generation_has_own_timeout(self):
        first, second, state = await self.run_two_turns(0.8)
        for events in (first, second):
            self.assertNotIn("error", [event for event, _, _ in events])
            self.assertIn("task_result", [event for event, _, _ in events])
            summary = next(data for event, _, data in events if event == "run_summary")
            self.assertEqual(summary["counts"]["rag"], 1)
            self.assertEqual(summary["counts"].get("web", 0), 0)
        self.assertEqual(state["asks"], 2)
        self.assertEqual(len(set(state["session_names"])), 2)
        self.assertEqual(len(set(state["deleted_ids"])), 2)

    async def test_slow_second_turn_reports_failure_without_reusing_previous_answer(self):
        first, second, state = await self.run_two_turns(0.05)
        self.assertIn("task_result", [event for event, _, _ in first])
        self.assertNotIn("task_result", [event for event, _, _ in second])
        self.assertNotIn("model_start", [event for event, _, _ in second])
        error = next(data for event, _, data in second if event == "error")
        self.assertEqual(error["error_code"], "RAG_TIMEOUT")
        summary = next(data for event, _, data in second if event == "run_summary")
        self.assertEqual(summary["counts"].get("models", 0), 0)
        self.assertEqual(state["asks"], 2)
        self.assertEqual(len(set(state["session_names"])), 2)


if __name__ == "__main__":
    unittest.main()
