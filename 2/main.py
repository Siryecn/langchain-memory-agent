# main.py - FastAPI 接口层（精简版，仅保留聊天核心）
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import logging

# 导入你提供的 Agent 核心类（确保 agent.py 和 main.py 在同一目录）
from agent import PersonalMemoryAgent

# ===================== 基础配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fastapi_agent")

# 初始化 FastAPI 应用
app = FastAPI(
    title="个人记忆智能助理API",
    description="对接个性化记忆Agent的FastAPI接口层",
    version="2.0.0"
)

# 配置 CORS 跨域（解决前端调用跨域问题）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境替换为具体前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== 全局 Agent 单例 =====================
try:
    agent = PersonalMemoryAgent()
    logger.info("✅ FastAPI 服务初始化成功：Agent 加载完成")
except Exception as e:
    logger.critical(f"❌ FastAPI 服务启动失败：Agent 初始化失败 - {str(e)}", exc_info=True)
    raise RuntimeError(f"服务启动失败：{str(e)}")

# ===================== Pydantic 请求模型 =====================
class ChatRequest(BaseModel):
    """对话接口请求模型（匹配前端传参）"""
    user_input: str  # 用户输入内容（必填）
    chat_history: Optional[List[Dict[str, str]]] = None  # 多轮对话历史（可选）

# ===================== 核心 API 接口 =====================
@app.get("/api/health", summary="服务健康检查", tags=["基础接口"])
async def health_check():
    """检查服务是否正常运行"""
    return {
        "code": 200,
        "status": "服务正常运行",
        "version": "2.0.0"
    }

@app.post("/api/chat", summary="核心对话接口", tags=["核心功能"])
async def chat_api(request: ChatRequest):
    """Agent 核心对话接口，支持多轮对话 + 自动记忆管理"""
    try:
        logger.info(f"接收对话请求：{request.user_input[:50]}...")
        result = agent.chat(
            user_input=request.user_input,
            chat_history=request.chat_history
        )
        return {
            "code": 200,
            "result": result
        }
    except Exception as e:
        logger.error(f"对话接口异常: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"对话处理失败：{str(e)}"
        )

# ===================== 服务启动入口 =====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app="main:app",
        host="0.0.0.0",
        port=7000,
        reload=True  # 开发环境保留，生产环境改 False
    )