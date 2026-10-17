from dotenv import load_dotenv,find_dotenv
import os
from langchain.chat_models import init_chat_model

# 加载配置文件
# find_dotenv() 确保找到 .env文件 递归查询当前项目文件夹
load_dotenv(find_dotenv())

# 控制单次模型回复的预算，避免多智能体的提示词和输出预留一起超过 TPM。
# 如需更长报告，可在 .env 中提高 LLM_MAX_OUTPUT_TOKENS。
max_output_tokens = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "2500"))
# 避免上游请求或 SDK 重试让后台任务长时间无反馈。
request_timeout_seconds = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "60"))
max_retries = int(os.getenv("LLM_MAX_RETRIES", "0"))
# GPT-5 会使用推理 token；默认使用最低推理强度以降低首轮工具规划的延迟和 TPM。
reasoning_effort = os.getenv("LLM_REASONING_EFFORT", "minimal")

model = init_chat_model(
    model=os.getenv("LLM_QWEN_MAX"),
    model_provider="openai",
    max_tokens=max_output_tokens,
    timeout=request_timeout_seconds,
    temperature=1.0, #控制模型输出的随机性，值越高输出越随机，值越低输出越确定
    max_retries=max_retries,
    reasoning_effort=reasoning_effort,
)
