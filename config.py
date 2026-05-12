# LLM Configuration
LLM_CONFIG = {
    "api_key": "425c0f7d-02a1-4a31-a027-416f087c3b31",
    "model_name": "doubao-seed-2-0-lite-260215",
    "base_url": "https://ark.cn-beijing.volces.com/api/v3",
    "temperature": 0.7,
    "max_tokens": 8192,
}

# Doubao-1.6-flash 备用1：425c0f7d-02a1-4a31-a027-416f087c3b31
# Doubao-1.6-flash 备用2：1665ac27-220b-4e4f-86d8-d6ca1ba151da

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

# 途牛 MCP CLI 配置
# 鉴权：TUNIU_API_KEY 走 .env 注入到子进程环境，本字典不放密钥
TUNIU_MCP_CONFIG = {
    # 可执行程序名或绝对路径；为 None 时由 tuniu_client 在 PATH 中解析
    "command": "tuniu",

    # 单次 CLI 调用超时（秒），与 tuniu --timeout 参数一致
    "timeout": 30,

    # 全局配额（覆盖 utils.tuniu_budget 单例的默认值）
    "rpm": 5,
    "rpd": 50,

    # 各工具缓存 TTL（秒）；价格波动大的设短，列表/详情稍长
    # 命中 key = (domain, tool, args)；不在表内的工具默认 0（不缓存）
    "cache_ttl": {
        "hotel:tuniu_hotel_search": 600,
        "hotel:tuniu_hotel_detail": 300,
        "flight:searchLowestPriceFlight": 180,
    },
}

# 高德地图 MCP Server 配置
AMAP_MCP_CONFIG = {
    # 高德地图 Web 服务 API Key - 在这里直接修改你申请的 Key
    # "AMAP_KEY": "1dd13742a147224131022165e14d6d55",
    "AMAP_KEY": "40e90e5245ec5f20bf578dff6fcad499",

    # 高德官方 MCP 服务 SSE 接入点（在线服务，无需本地启动）
    "sse_endpoint": "https://mcp.amap.com/sse",
    
    # 本地开发备选方案（需要 Node.js 和 npx）
    # 使用本地方式时，执行: npm install -g @amap/amap-maps-mcp-server
    # "local_mode": True,  # 改为 True 启用本地模式
    # "command": "npx",
    # "args": ["-y", "@amap/amap-maps-mcp-server"],
    
    "timeout": 30,
}