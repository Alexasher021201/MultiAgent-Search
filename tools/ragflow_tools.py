#  get_assistant_list 获取聊天助手和知识库信息
#  create_ask_delete  创建提问和删除会话获取rag查询结果
from langchain_core.tools import tool
import re
# 导入依赖
from ragflow_sdk import RAGFlow #链接rag服务的客户端

from api.monitor import monitor
from rawflow.rag_config import _load_ragflow_env

# 创建一个ragflow的客户端
api_key , base_url =_load_ragflow_env()
ragflow_client = RAGFlow(api_key=api_key,base_url=base_url)



# 1. 查询现在知识库中有哪些聊天助手和对应知识库的信息 （方便我们知道rag可以给我们提供哪些数据）
@tool
def get_assistant_list() -> str:
    """
    调用此工具，可以查询ragflow服务器中有哪些助手和助手关联的知识库信息！
    供模型参考，可以从哪个助手获取对应的内部文档信息！
    强调：想向某个助手提问，必须想要调用此工具查询助手的信息和名称
    返回结果： 有-> 名称:助手名称,助手描述：xxxx,关联的知识库：知识库的名、知识库的名字、
             没有 -> 没有任何可用助手
             异常 -> 查询助手信息异常，无可用助手
    :return:
    """

    # 埋点,调用工具了告诉前端哪个工具被调用了！！
    monitor.report_tool(tool_name="ragflow聊天助手列表查询工具：get_assistant_list")

    # 1. 创建ragflow客户端
    try:
        # 2. ragflow客户端查询所有的聊天助手 page: int = 1, page_size: int = 30
        chat_list = ragflow_client.list_chats()
        if not chat_list:
            return "没有任何可用助手"
        # 3. 查询聊天助手的知识库信息
        count_chat_info = "" #存储所有会话信息
        for chat in chat_list:
            dataset_names = []
            dataset_list = chat.datasets #当前聊天助手关联的知识库
            if dataset_list and isinstance(dataset_list,list):
                # 知识库的name
                for dataset in dataset_list:
                    # print(dataset)
                    dataset_names.append(dataset['name']) # 将一个助手的知识库的名字加入到列表中

            # 拼接格式: 助手名称 + 介绍 + 关联知识库列表
            count_chat_info += f"助手名称:{chat.name};功能介绍：{chat.description}; 关联的知识库：{'、'.join(dataset_names)} \n"
        return count_chat_info
    except Exception as e:
        return f"查询助手信息异常，无可用助手,异常信息:{str(e)}"

def _retrieval_question(question: str) -> str:
    """提取明确标注的查询问题，防止把委派指令传给检索引擎。"""
    question = question.strip()
    match = re.search(r'查询并回答\s*[:：]\s*["“](.+?)["”]\s*要求\s*[:：]', question, re.DOTALL)
    return match.group(1).strip() if match else question


# 2. 对某个助手进行提问（创建会话 -》 提问 -》 删除会话）
@tool
def create_ask_delete(chat_name: str, question: str) -> str:
    """
    想某个助手，创建单次会话进行提问，提问完毕以后会关闭会话！
    主要查询ragflow中相关的信息！
    注意：调用此工具之前，必须先调用 get_assistant_list工具明确查询助手的名字和对应的问题
    :param chat_name: 助手的名字！上一个工具get_assistant_list告诉大模型的只有名字
    :param question: 只传用户要查的简短问题，例如“退款金额超过500元由谁审批？”。
        不要传整段委派任务、读取文件指令、输出格式、版本比较等要求。
    :return: 返回提问的结果
    """
    # 埋点,调用工具了告诉前端哪个工具被调用了！！
    question = _retrieval_question(question)
    monitor.report_tool(tool_name="ragflow提问助手工具：create_ask_delete",args={"chat_name":chat_name,"question":question})
    print(f"[RAGFlow] assistant={chat_name!r}, question={question!r}")
    # 1. 创建ragflow客户端
    # 2. 查询对应name的chat
    use_chat = None
    session = None
    try:
        chats = ragflow_client.list_chats(name=chat_name)
        matches = [chat for chat in chats if chat.name == chat_name]
        if len(matches) != 1:
            return "提问失败：助手不存在或名称重复，请重新调用 get_assistant_list 确认助手名称。"
        use_chat = matches[0]
        # 3. chat上创建一个会话
        session = use_chat.create_session(name="temp_session_ask")
        # 4. 使用会话进行提问
        # 返回的提问结果是流式
        response = session.ask(question = question,stream=True)
        # 接收总结果
        result = ""
        # 流的每一部分的对象 part
        for part in response:
            # 数据存在对象中content上！！
            # print(part.content)
            result = part.content
        print(f"[RAGFlow] answer={result!r}")
        return result
    except Exception as e:
        print(f"[RAGFlow] query failed: {e}")
        return f"提问失败，错误原因：{str(e)}"
    finally:
        if use_chat is not None and session is not None:
            try:
                use_chat.delete_sessions(ids=[session.id])
            except Exception as e:
                print(f"[RAGFlow] 临时会话清理失败：{e}")

# if __name__ == '__main__':
#     # print(get_assistant_list())
#     # print(create_ask_delete("你的助手名称", "你的问题"))
