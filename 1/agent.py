from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
import requests

# 必选：APIkey配置
API_KEY1 = "your-openweathermap-api-key"   # OpenWeatherMap 天气 API Key
API_KEY2 = "your-dashscope-api-key"        # 阿里云 DashScope API Key

llm = ChatOpenAI(
    model="qwen3.6-plus",
    api_key=API_KEY2,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# @tool + docstring
@tool
def get_weather(city: str) -> str:
    """用于查询任意城市的实时天气，输入为城市名称"""  # 仅新增这行docstring
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY1}&lang=zh&units=metric"
        res = requests.get(url).json()
        return f"{res['name']}当前温度：{res['main']['temp']}摄氏度，天气：{res['weather'][0]['description']}"
    except:
        return "天气查询失败，请检查城市名称"

# @tool + docstring
@tool
def write_weekly(work_content: str) -> str:
    """用于根据工作内容生成职场周报，输入为具体工作描述"""  # 仅新增这行docstring
    tishi = f"请根据以下工作内容，生成一份简洁的职场周报，分工作完成、问题与改进、下周计划三部分：{work_content}"
    return llm.invoke(tishi).content

tools = [get_weather, write_weekly]

prompt = ChatPromptTemplate.from_messages([
    ("system","你是一个高效的办公提效小助手，会根据用户需求，选择合适的工具完成任务，工具处理后直接整理结果回复用户"),
    ("user","{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent= create_tool_calling_agent(llm,tools,prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({"input":"帮我查一下ChengDu的天气，我今天完成了后端IM系统的部署，给我写一份周报"})
print(f"最终的结果是：{result['output']}")