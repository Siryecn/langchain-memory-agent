import os
import threading
import logging
import re
from typing import Optional, List, Union, Dict, Any
from dotenv import load_dotenv
from langchain_core.tools import tool
# 兼容阿里云的OpenAI导入（保留）
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import requests
# 阿里云SDK保留（必须），但无硬编逻辑
import dashscope
from dashscope import TextEmbedding

# ===================== 生产环境配置与日志 =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# load_dotenv()将.env临时写入到系统环境变量中，os.getenv将系统环境变量里的变量对应的值取出
load_dotenv()


# 🔥 配置类：彻底删除所有阿里云硬编默认值，100%从.env读取
class Config:
    # 阿里云核心配置：无任何硬编默认值，必须从.env配置
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")  # 阿里云dashscope密钥
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL")  # 阿里云兼容地址
    OPENAI_MODEL_NAME: str = os.getenv("OPENAI_MODEL_NAME")  # 阿里云模型名（qwen-plus等）
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL")  # 阿里云嵌入模型

    # 业务配置：仅保留本地目录/阈值的合理默认值（非阿里云业务参数）
    WEATHER_API_KEY: str = os.getenv("WEATHER_API_KEY")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_memory")
    VECTOR_SEARCH_K: int = int(os.getenv("VECTOR_SEARCH_K", "3"))
    VECTOR_SCORE_THRESHOLD: float = float(os.getenv("VECTOR_SCORE_THRESHOLD", "0.7"))
    AGENT_MAX_ITERATIONS: int = int(os.getenv("AGENT_MAX_ITERATIONS", "10"))  # 提高迭代次数支持批量操作
    AGENT_MAX_EXECUTION_TIME: float = float(os.getenv("AGENT_MAX_EXECUTION_TIME", "30"))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "30"))

    @classmethod
    def validate(cls) -> None:
        """校验阿里云必须配置的参数，无硬编逻辑"""
        required_keys = [
            "OPENAI_API_KEY", "OPENAI_BASE_URL",
            "OPENAI_MODEL_NAME", "OPENAI_EMBEDDING_MODEL"
        ]
        for key in required_keys:       # if not key:只能判断前面已经被真实赋值的。key = "OPENAI_API_KEY"，判断的是这个字符串
            if not getattr(cls, key):
                raise ValueError(f"生产环境启动失败：请在.env文件中配置 {key}")
        # 初始化阿里云SDK（从.env读密钥，无硬编）
        dashscope.api_key = cls.OPENAI_API_KEY


Config.validate()


# ===================== 阿里云原生Embedding类（无硬编） =====================
class DashScopeEmbeddings(Embeddings):
    """阿里云dashscope原生Embedding，无硬编默认值，从Config读模型名"""

    def __init__(self, model: str):  # 去掉硬编默认值，必须传Config里的模型名
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            response = TextEmbedding.call(
                model=self.model,
                input=texts,
                text_type="document"
            )
            return [item["embedding"] for item in response.output["embeddings"]]
        except Exception as e:
            logger.error(f"Embedding生成失败: {str(e)}", exc_info=True)
            raise

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


# ===================== 生产级个性化记忆 Agent =====================
class PersonalMemoryAgent:
    def __init__(self):
        logger.info("正在初始化个性化记忆 Agent...")
        self.vector_store_lock = threading.Lock()
        self._init_llm()
        self._init_vector_store()
        self.tools = self._init_tools()
        self._init_agent()
        logger.info("个性化记忆 Agent 初始化完成！")

    def _init_llm(self) -> None:
        """初始化阿里云LLM+Embedding，无硬编，全读Config"""
        logger.info(f"正在初始化 LLM 模型：{Config.OPENAI_MODEL_NAME}")
        # LLM初始化：全读Config，无硬编
        self.llm = ChatOpenAI(
            model=Config.OPENAI_MODEL_NAME,
            api_key=Config.OPENAI_API_KEY,
            base_url=Config.OPENAI_BASE_URL,
            temperature=Config.LLM_TEMPERATURE,
            timeout=Config.LLM_TIMEOUT,
            max_retries=2
        )

        logger.info(f"正在初始化 Embeddings 模型：{Config.OPENAI_EMBEDDING_MODEL}")
        # Embedding初始化：传Config里的模型名，无硬编
        self.embeddings = DashScopeEmbeddings(
            model=Config.OPENAI_EMBEDDING_MODEL
        )

    def _init_vector_store(self) -> None:
        # 👇 完全复用原有逻辑，无修改
        logger.info(f"正在初始化向量数据库，持久化目录：{Config.CHROMA_PERSIST_DIR}")
        self.vector_store = Chroma(
            collection_name="personal_memory",
            embedding_function=self.embeddings,
            persist_directory=Config.CHROMA_PERSIST_DIR
        )

        self.retriever: VectorStoreRetriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": Config.VECTOR_SEARCH_K, "score_threshold": Config.VECTOR_SCORE_THRESHOLD}
        )

    def _init_agent(self) -> None:
        # 👇 核心修改：更新system prompt，告知Agent支持灵活的添加/删除操作
        logger.info("正在初始化 Agent 执行器...")

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是具备灵活记忆管理能力的个性化智能助理，严格遵循以下规则：
            1. 核心身份：你是用户的专属记忆管家，支持灵活的记忆添加/删除/查询操作。
            2. 记忆添加规则（灵活支持）：
               - 用户单条添加：如"添加记忆：我家在上海"，直接调用add_memory添加。
               - 用户批量添加：如"添加记忆：1.我家在上海 2.每天学习3小时 3.每周五运动"或"添加记忆：我家在上海、每天学习3小时"，自动拆分多条记忆并批量添加。
               - 自然语言提及的个性化信息，也可主动询问是否添加（非强制）。
            3. 记忆删除规则（灵活支持）：
               - 精准单条删除：如"删除记忆：我家在上海"，精准匹配内容删除。
               - 精准批量删除：如"删除记忆：我家在上海、每天学习3小时"，拆分后批量精准删除。
               - 关键词批量删除：如"批量删除和上海相关的记忆"，删除所有含指定关键词的记忆。
               - 序号删除：如"删除第1、3条记忆"，先查询所有记忆，按序号匹配删除对应内容。
            4. 工具使用规则：
               - 添加记忆优先调用add_memory工具，支持批量传入列表。
               - 删除记忆优先调用manage_memory工具，根据用户需求选择对应删除方式（精准单条/批量、关键词批量、序号删除）。
               - 执行完成后清晰告知用户操作结果（如添加了几条、删除了几条），禁止暴露工具调用细节。
            5. 安全规则：禁止编造记忆，所有操作基于用户提供的内容，删除前可简要确认（可选）。
            6. 异常处理：操作失败时返回友好提示，如"未找到该记忆"、"批量添加失败，请检查格式"。
            7. 任务处理规则：当用户要求生成周报时，请直接调用write_weekly_report工具，并将用户提供的工作内容作为参数传入，无需调用其他工具。"""),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=False,
            handle_parsing_errors="非常抱歉，我暂时无法处理这个请求，请你换一种方式描述",
            max_iterations=Config.AGENT_MAX_ITERATIONS,
            max_execution_time=Config.AGENT_MAX_EXECUTION_TIME
        )

    def _query_memory(self, keyword: str) -> str:
        # 👇 完全复用原有逻辑，无修改
        try:
            with self.vector_store_lock:
                docs = self.retriever.invoke(keyword)
            if not docs:
                return ""
            return "\n".join([f"- {doc.page_content}" for doc in docs])
        except Exception as e:
            logger.error(f"查询记忆失败: {str(e)}", exc_info=True)
            return ""

    # ===================== 核心修改1：灵活的添加记忆方法（支持单条/批量） =====================
    def _add_memory(self, memory_content: Union[str, List[str]]) -> str:
        try:
            with self.vector_store_lock:
                # 处理批量添加（列表形式）
                if isinstance(memory_content, list):
                    # 过滤空内容
                    valid_memories = [mem.strip() for mem in memory_content if mem.strip()]
                    if not valid_memories:
                        return "❌ 批量添加失败：没有有效记忆内容"
                    # 批量添加
                    docs = [Document(page_content=mem) for mem in valid_memories]
                    self.vector_store.add_documents(docs)
                    logger.info(f"批量添加记忆成功，共{len(valid_memories)}条：{valid_memories}")
                    return f"✅ 批量添加记忆成功！共添加{len(valid_memories)}条：\n" + "\n".join(
                        [f"- {mem}" for mem in valid_memories])
                # 处理单条添加（字符串形式）
                else:
                    content = memory_content.strip()
                    if not content:
                        return "❌ 添加失败：记忆内容不能为空"
                    doc = Document(page_content=content)
                    self.vector_store.add_documents([doc])
                    logger.info(f"单条添加记忆成功：{content}")
                    return f"✅ 已成功添加记忆：{content}"
        except Exception as e:
            logger.error(f"添加记忆失败: {str(e)}", exc_info=True)
            return f"❌ 记忆添加失败：{str(e)[:50]}"

    # ===================== 核心修改2：灵活的管理记忆方法（支持多种删除方式） =====================
    def _manage_memory(self, operation: str, keyword: Optional[str] = None) -> str:
        try:
            with self.vector_store_lock:
                # 查询所有记忆（基础功能）
                if operation == "查询所有":
                    all_data = self.vector_store.get()
                    all_memories = all_data["documents"]
                    if not all_memories:
                        return "📭 你的记忆库目前为空"
                    # 带序号返回，方便用户按序号删除
                    numbered_memories = [f"{i + 1}. {mem}" for i, mem in enumerate(all_memories)]
                    logger.info("用户查询了所有记忆")
                    return "📋 你的所有记忆（带序号）：\n" + "\n".join(numbered_memories)

                # 关键词查询（基础功能）
                elif operation == "查询":
                    if not keyword:
                        return "❌ 查询失败：请输入查询关键词"
                    memory_content = self._query_memory(keyword)
                    if not memory_content:
                        return f"🔍 未找到与「{keyword}」相关的记忆"
                    logger.info(f"用户查询记忆：{keyword}")
                    return f"🔍 找到与「{keyword}」相关的记忆：\n{memory_content}"

                # 灵活删除（核心修改）
                elif operation == "删除":
                    if not keyword:
                        return "❌ 删除失败：请输入要删除的内容/关键词/序号"

                    # 方式1：按序号删除（如"1,3"、"第1、3条"）
                    if re.search(r"\d+", keyword):
                        # 提取所有数字序号
                        indices = re.findall(r"\d+", keyword)
                        target_indices = [int(idx) - 1 for idx in indices if int(idx) - 1 >= 0]
                        all_data = self.vector_store.get()
                        all_ids = all_data["ids"]
                        all_memories = all_data["documents"]

                        delete_ids = []
                        delete_contents = []
                        for idx in target_indices:
                            if idx < len(all_memories):
                                delete_ids.append(all_ids[idx])
                                delete_contents.append(all_memories[idx])

                        if not delete_ids:
                            return "❌ 未找到对应序号的记忆"
                        self.vector_store.delete(ids=delete_ids)
                        logger.info(f"按序号删除记忆：{delete_contents}")
                        return f"✅ 已删除指定序号的记忆：\n" + "\n".join([f"- {mem}" for mem in delete_contents])

                    # 方式2：批量精准删除（内容用顿号/逗号/换行分隔）
                    elif "、" in keyword or "," in keyword or "\n" in keyword:
                        # 拆分批量删除的内容
                        split_chars = ["、", ",", "\n"]
                        memories_to_delete = keyword
                        for char in split_chars:
                            memories_to_delete = memories_to_delete.replace(char, "|")
                        memory_list = [mem.strip() for mem in memories_to_delete.split("|") if mem.strip()]

                        all_data = self.vector_store.get()
                        all_ids = all_data["ids"]
                        all_memories = all_data["documents"]

                        delete_ids = []
                        delete_contents = []
                        for mem in memory_list:
                            for i, content in enumerate(all_memories):
                                if content.strip() == mem.strip():
                                    delete_ids.append(all_ids[i])
                                    delete_contents.append(content)

                        if not delete_ids:
                            return f"❌ 未找到要删除的记忆：{memory_list}"
                        self.vector_store.delete(ids=delete_ids)
                        logger.info(f"批量精准删除记忆：{delete_contents}")
                        return f"✅ 已批量删除指定记忆：\n" + "\n".join([f"- {mem}" for mem in delete_contents])

                    # 方式3：关键词批量删除
                    elif "批量删除" in keyword or "所有" in keyword:
                        # 提取关键词（去掉"批量删除"、"所有"、"相关的"等前缀）
                        keyword_clean = re.sub(r"批量删除|所有|相关的|的记忆|关于", "", keyword).strip()
                        docs_with_score = self.vector_store.similarity_search_with_score(keyword_clean, k=20)
                        if not docs_with_score:
                            return f"❌ 未找到与「{keyword_clean}」相关的记忆"

                        delete_ids = [doc[0].id for doc in docs_with_score if doc[0].id is not None]
                        delete_contents = [doc[0].page_content for doc in docs_with_score]

                        self.vector_store.delete(ids=delete_ids)
                        logger.info(f"关键词批量删除记忆：{delete_contents}")
                        return f"✅ 已删除所有与「{keyword_clean}」相关的记忆（共{len(delete_contents)}条）：\n" + "\n".join(
                            [f"- {mem}" for mem in delete_contents])

                    # 方式4：精准单条删除
                    else:
                        all_data = self.vector_store.get()
                        all_ids = all_data["ids"]
                        all_memories = all_data["documents"]

                        delete_ids = []
                        delete_content = ""
                        for i, content in enumerate(all_memories):
                            if content.strip() == keyword.strip():
                                delete_ids.append(all_ids[i])
                                delete_content = content

                        if not delete_ids:
                            return f"❌ 未找到这条记忆：{keyword}"
                        self.vector_store.delete(ids=delete_ids)
                        logger.info(f"精准删除记忆：{delete_content}")
                        return f"✅ 已精准删除记忆：{delete_content}"

                else:
                    return "❌ 无效操作！支持的操作：查询所有、查询（关键词）、删除（内容/序号/关键词）"
        except Exception as e:
            logger.error(f"管理记忆失败: {str(e)}", exc_info=True)
            return f"❌ 记忆管理失败：{str(e)[:50]}"

    # ===================== 核心修改3：更新工具定义，适配灵活操作 =====================
    def _init_tools(self):
        llm = self.llm

        @tool
        def get_weather(city: Optional[str] = None) -> str:
            """查询指定城市的当前天气信息。如果未指定城市，会先尝试从用户记忆中查询常住城市。"""
            if not city:
                memory_content = self._query_memory("常住城市 居住地址 家住城市")
                if memory_content:
                    city = memory_content.split("\n")[0].strip().lstrip("- ").strip()
                else:
                    return "我还不知道你的常住城市，请告诉我你要查询哪个城市的天气（示例：帮我查北京的天气）"

            try:
                if not Config.WEATHER_API_KEY:
                    return "天气查询功能未配置密钥，请先在.env文件中配置WEATHER_API_KEY"

                url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={Config.WEATHER_API_KEY}&units=metric&lang=zh_cn"
                response = requests.get(url, timeout=10).json()

                if response.get("cod") != 200:
                    error_msg = response.get('message', '未知错误')
                    if "Invalid API key" in error_msg:
                        logger.error("天气查询失败：API密钥无效")
                        return "天气查询失败：你的API密钥无效，请检查.env文件中的WEATHER_API_KEY是否正确"
                    elif "city not found" in error_msg:
                        return f"未找到「{city}」这个城市，请检查城市名称是否正确"
                    else:
                        logger.error(f"天气查询失败: {error_msg}")
                        return f"天气查询失败：{error_msg}"

                city_name = response.get("name", city)
                temp = response['main']['temp']
                weather_desc = response['weather'][0]['description']
                humidity = response['main']['humidity']
                wind_speed = response['wind']['speed']
                logger.info(f"成功查询天气: {city_name}")
                return f"{city_name}当前天气：{weather_desc}，温度：{temp}℃，湿度：{humidity}%，风速：{wind_speed}m/s"

            except requests.exceptions.RequestException as e:
                logger.error(f"天气查询网络错误: {str(e)}", exc_info=True)
                return "天气查询网络错误：请检查你的网络连接是否正常"
            except Exception as e:
                logger.error(f"天气查询失败: {str(e)}", exc_info=True)
                return "天气查询失败：暂时无法获取天气信息，请稍后重试"

        @tool
        def write_weekly_report(work_content: str) -> str:
            """根据用户提供的核心工作内容，结合用户记忆中的学习/工作记录，生成一份规范的学习/工作周报。"""
            memory_content = self._query_memory("学习计划 学习时间 学习内容 工作内容 工作计划")
            prompt = f"""
            请根据以下核心工作内容，生成规范、详实、贴合实际的学习/工作周报，严格分为「本周完成」「问题与改进」「下周计划」三个部分。
            可结合下方的用户个性化记录补充细节，内容要具体、不空泛，符合用户的实际情况。
            【用户个性化记忆记录】：{memory_content if memory_content else '无额外记忆'}
            【本周核心工作/学习内容】：{work_content}
            """
            logger.info("正在生成周报...")
            return self.llm.invoke(prompt).content

        @tool
        def add_memory(memory_content: Union[str, List[str]]) -> str:
            """添加记忆，支持单条字符串或批量列表形式"""
            return self._add_memory(memory_content)

        @tool
        def manage_memory(operation: str, keyword: Optional[str] = None) -> str:
            """管理记忆，支持：
            - 查询所有：operation="查询所有"，keyword=None
            - 查询：operation="查询"，keyword=查询关键词
            - 删除：operation="删除"，keyword=要删除的内容/序号/关键词
            """
            return self._manage_memory(operation, keyword)

        return [get_weather, write_weekly_report, add_memory, manage_memory]

    def chat(self, user_input: str,
             chat_history: Optional[Union[List[BaseMessage], List[Dict[str, str]]]] = None) -> str:
        # 👇 完全复用原有逻辑，无修改
        logger.info(f"收到用户输入: {user_input[:50]}...")

        chat_history = chat_history or []
        formatted_history = []

        for msg in chat_history:
            if isinstance(msg, dict):
                if msg["role"] == "user":
                    formatted_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    formatted_history.append(AIMessage(content=msg["content"]))
            elif isinstance(msg, BaseMessage):
                formatted_history.append(msg)

        try:
            result = self.agent_executor.invoke({
                "input": user_input,
                "chat_history": formatted_history
            })
            response = result["output"]
            logger.info(f"Agent 回复完成: {response[:50]}...")
            return response
        except Exception as e:
            logger.error(f"Agent 执行失败: {str(e)}", exc_info=True)
            return "非常抱歉，我暂时无法处理这个请求，请你换一种方式描述"

    def __del__(self):
        # 👇 完全复用原有逻辑，无修改
        try:
            if hasattr(self, "vector_store"):
                logger.info("正在安全关闭向量数据库连接...")
                del self.vector_store
        except Exception as e:
            logger.warning(f"关闭向量数据库时发生警告: {str(e)}")


# ===================== 生产环境测试入口 =====================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("个性化记忆 Agent 生产环境测试启动")
    logger.info("=" * 60)

    try:
        agent = PersonalMemoryAgent()
        chat_history = []

        # 测试1：批量添加记忆
        print("\n【测试1：批量添加记忆】")
        res1 = agent.chat("添加记忆：1.我家在ShangHai 2.每天学习3小时 3.每周五晚上运动", chat_history)
        print(res1)
        chat_history.extend([
            {"role": "user", "content": "添加记忆：1.我家在ShangHai 2.每天学习3小时 3.每周五晚上运动"},
            {"role": "assistant", "content": res1}
        ])

        # 测试2：查询所有记忆（带序号）
        print("\n【测试2：查询所有记忆】")
        res2 = agent.chat("查询所有记忆", chat_history)
        print(res2)
        chat_history.extend([
            {"role": "user", "content": "查询所有记忆"},
            {"role": "assistant", "content": res2}
        ])

        # 测试3：按序号删除记忆
        print("\n【测试3：按序号删除记忆】")
        res3 = agent.chat("删除第1、3条记忆", chat_history)
        print(res3)
        chat_history.extend([
            {"role": "user", "content": "删除第1、3条记忆"},
            {"role": "assistant", "content": res3}
        ])

        # 测试4：关键词批量删除
        print("\n【测试4：关键词批量删除】")
        res4 = agent.chat("批量删除和学习相关的记忆", chat_history)
        print(res4)
        chat_history.extend([
            {"role": "user", "content": "批量删除和学习相关的记忆"},
            {"role": "assistant", "content": res4}
        ])

        logger.info("=" * 60)
        logger.info("个性化记忆 Agent 生产环境测试完成！")
        logger.info("=" * 60)

    except Exception as e:
        logger.critical(f"生产环境测试启动失败: {str(e)}", exc_info=True)
        raise