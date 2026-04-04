"""
Configuration for the Aligo Multi-Agent System
"""

# LLM Configuration
LLM_CONFIG = {
    "api_key": "1665ac27-220b-4e4f-86d8-d6ca1ba151da",
    "model_name": "doubao-seed-1-6-flash-250828",
    "base_url": "https://ark.cn-beijing.volces.com/api/v3",
    "temperature": 0.7,
    "max_tokens": 8192,
}

# System Configuration
SYSTEM_CONFIG = {
    "enable_llm": True,  # Set to True to use LLM (recommended), False for rule-based
    "log_level": "INFO",
    "max_retries": 3,
    "timeout": 60,  # Increased timeout for better stability
}

# RAG 知识库：嵌入模型（本地路径，无需连 HuggingFace）
RAG_CONFIG = {
    "embedding_model": "data/models/bge-small-zh-v1.5",
}

# 连接与可用性：重试、熔断、健康检查
RESILIENCE_CONFIG = {
    "max_retries": 3,              # 单次请求最大重试次数（与 SYSTEM_CONFIG 对齐）
    "retry_base_delay_sec": 1.0,   # 重试退避基数（秒）
    "retry_max_delay_sec": 30.0,   # 重试退避上限（秒）
    "circuit_failure_threshold": 5, # 连续失败多少次后熔断
    "circuit_recovery_timeout_sec": 60.0,  # 熔断后多少秒进入半开
    "circuit_half_open_successes": 2,      # 半开状态下连续成功多少次后关闭
    "health_check_timeout_sec": 10.0,      # 健康检查请求超时（秒）
}


# RollingGo 酒店 MCP Server 配置
ROLLINGGO_MCP_CONFIG = {
    # 你申请到的 API Key（填入实际值）
    "ROLLINGGO_API_KEY": "mcp_9dd23be789524ab0a9bdbd9f8def827a",

    # 启动方式：pip install rollinggo-mcp 后直接调用（使用完整路径避免 PATH 问题）
    "command": r"C:\Users\shu\anaconda3\envs\grad_pro\Scripts\rollinggo-mcp.exe",
    "args": [],
    # 备用方式（需要 Node.js）：
    # "command": "npx",
    # "args": ["-y", "rollinggo-mcp"],

    # 超时设置（秒）
    "timeout": 30,

    # 默认搜索参数
    "default_size": 5,           # 默认返回酒店数量
    "default_currency": "CNY",   # 默认货币
    "default_country": "CN",     # 默认国家码
}

# 航班 MCP Server 配置（Streamable HTTP）
# 认证方式：API Key 已内嵌于 URL 的 ?api_key= 参数中，无需额外请求头
FLIGHT_MCP_CONFIG = {
    "url": "https://ai.variflight.com/servers/aviation/mcp/?api_key=sk-xn0QkmUlz1RqlJfoL0b65kjlVKXYu01PViLtWfVA1p0",
}

# 高德地图 MCP Server 配置
AMAP_MCP_CONFIG = {
    # 高德地图 Web 服务 API Key - 在这里直接修改你申请的 Key
    "AMAP_KEY": "1dd13742a147224131022165e14d6d55",
    
    # 高德官方 MCP 服务 SSE 接入点（在线服务，无需本地启动）
    "sse_endpoint": "https://mcp.amap.com/sse",
    
    # 本地开发备选方案（需要 Node.js 和 npx）
    # 使用本地方式时，执行: npm install -g @amap/amap-maps-mcp-server
    # "local_mode": True,  # 改为 True 启用本地模式
    # "command": "npx",
    # "args": ["-y", "@amap/amap-maps-mcp-server"],
    
    "timeout": 30,
}