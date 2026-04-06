"""
初始化RAG知识库 (Plugin Version)
从 .claude/skills/ask-question/data/documents 目录加载旅游相关文档并导入到向量数据库中
"""
import sys
import os
import re
import importlib.util
from pathlib import Path
from typing import List, Dict

# ============================================================
# 城市结构化元数据：供地图 API 工具调用直接提取参数
# coordinates: [经度, 纬度]  city_tier: 城市等级  avg_days: 建议游玩天数
# ============================================================
CITY_METADATA: Dict[str, Dict] = {
    "重庆":   {"coordinates": [106.5516, 29.5630], "city_tier": 1, "avg_days": 3, "region": "西南",  "tags": ["山城", "火锅", "夜景", "网红打卡"]},
    "北京":   {"coordinates": [116.4074, 39.9042], "city_tier": 1, "avg_days": 4, "region": "华北",  "tags": ["历史文化", "古迹", "皇家园林", "政治中心"]},
    "上海":   {"coordinates": [121.4737, 31.2304], "city_tier": 1, "avg_days": 3, "region": "华东",  "tags": ["国际都市", "美食", "外滩", "购物"]},
    "成都":   {"coordinates": [104.0665, 30.5728], "city_tier": 1, "avg_days": 3, "region": "西南",  "tags": ["熊猫", "火锅", "休闲", "川菜"]},
    "西安":   {"coordinates": [108.9398, 34.3416], "city_tier": 1, "avg_days": 3, "region": "西北",  "tags": ["古都", "兵马俑", "历史", "美食"]},
    "杭州":   {"coordinates": [120.1551, 30.2741], "city_tier": 1, "avg_days": 2, "region": "华东",  "tags": ["西湖", "江南", "茶文化", "互联网"]},
    "广州":   {"coordinates": [113.2644, 23.1291], "city_tier": 1, "avg_days": 2, "region": "华南",  "tags": ["粤菜", "早茶", "商贸", "岭南文化"]},
    "深圳":   {"coordinates": [114.0579, 22.5431], "city_tier": 1, "avg_days": 2, "region": "华南",  "tags": ["科技", "现代都市", "主题乐园", "创新"]},
    "武汉":   {"coordinates": [114.3054, 30.5931], "city_tier": 1, "avg_days": 2, "region": "华中",  "tags": ["江城", "热干面", "樱花", "大学城"]},
    "长沙":   {"coordinates": [112.9388, 28.2278], "city_tier": 2, "avg_days": 2, "region": "华中",  "tags": ["网红美食", "湖南菜", "橘子洲", "夜生活"]},
    "南京":   {"coordinates": [118.7969, 32.0603], "city_tier": 1, "avg_days": 2, "region": "华东",  "tags": ["六朝古都", "历史", "秦淮河", "民国建筑"]},
    "苏州":   {"coordinates": [120.5853, 31.2990], "city_tier": 2, "avg_days": 2, "region": "华东",  "tags": ["园林", "江南水乡", "丝绸", "昆曲"]},
    "三亚":   {"coordinates": [109.5122, 18.2523], "city_tier": 2, "avg_days": 4, "region": "华南",  "tags": ["海滩", "热带度假", "海鲜", "潜水"]},
    "昆明":   {"coordinates": [102.7123, 25.0406], "city_tier": 2, "avg_days": 2, "region": "西南",  "tags": ["春城", "花卉", "云南美食", "石林"]},
    "天津":   {"coordinates": [117.1903, 39.1255], "city_tier": 1, "avg_days": 2, "region": "华北",  "tags": ["相声", "狗不理", "租界建筑", "海河"]},
    "青岛":   {"coordinates": [120.3826, 36.0671], "city_tier": 2, "avg_days": 2, "region": "华东",  "tags": ["啤酒", "海鲜", "德式建筑", "海岸线"]},
    "桂林":   {"coordinates": [110.2991, 25.2736], "city_tier": 2, "avg_days": 3, "region": "华南",  "tags": ["山水", "漓江", "喀斯特地貌", "阳朔"]},
    "大理白族自治州": {"coordinates": [100.2250, 25.6065], "city_tier": 2, "avg_days": 3, "region": "西南", "tags": ["古城", "洱海", "白族文化", "慢生活"]},
    "厦门":   {"coordinates": [118.0894, 24.4798], "city_tier": 2, "avg_days": 2, "region": "华东",  "tags": ["鼓浪屿", "闽南文化", "海鲜", "文艺"]},
    "郑州":   {"coordinates": [113.6254, 34.7466], "city_tier": 2, "avg_days": 2, "region": "华中",  "tags": ["少林寺", "黄河", "中原文化", "交通枢纽"]},
    "济南":   {"coordinates": [117.0009, 36.6758], "city_tier": 2, "avg_days": 2, "region": "华东",  "tags": ["泉城", "趵突泉", "大明湖", "鲁菜"]},
    "贵阳":   {"coordinates": [106.6302, 26.6477], "city_tier": 2, "avg_days": 2, "region": "西南",  "tags": ["凉爽避暑", "苗族文化", "酸汤鱼", "大数据"]},
    "南宁":   {"coordinates": [108.3665, 22.8170], "city_tier": 2, "avg_days": 2, "region": "华南",  "tags": ["绿城", "壮族文化", "东南亚风情", "老友粉"]},
    "福州":   {"coordinates": [119.2965, 26.0745], "city_tier": 2, "avg_days": 2, "region": "华东",  "tags": ["三坊七巷", "温泉", "闽菜", "海峡城市"]},
    "合肥":   {"coordinates": [117.2272, 31.8206], "city_tier": 2, "avg_days": 1, "region": "华东",  "tags": ["科教城市", "巢湖", "徽派文化", "高新技术"]},
    "大连":   {"coordinates": [121.6147, 38.9140], "city_tier": 2, "avg_days": 2, "region": "东北",  "tags": ["海滨城市", "足球", "海鲜", "俄式建筑"]},
    "哈尔滨": {"coordinates": [126.5358, 45.8038], "city_tier": 2, "avg_days": 3, "region": "东北",  "tags": ["冰雪节", "俄式建筑", "东北美食", "中央大街"]},
    "洛阳":   {"coordinates": [112.4539, 34.6197], "city_tier": 2, "avg_days": 2, "region": "华中",  "tags": ["牡丹", "龙门石窟", "古都", "水席"]},
    "遵义":   {"coordinates": [106.9272, 27.7254], "city_tier": 3, "avg_days": 2, "region": "西南",  "tags": ["红色文化", "遵义会议", "茅台酒", "喀斯特"]},
    "烟台":   {"coordinates": [121.3914, 37.5396], "city_tier": 3, "avg_days": 2, "region": "华东",  "tags": ["苹果", "葡萄酒", "海鲜", "蓬莱仙境"]},
    "威海":   {"coordinates": [122.1198, 37.5134], "city_tier": 3, "avg_days": 2, "region": "华东",  "tags": ["海滨度假", "刘公岛", "海鲜", "宜居城市"]},
    "南昌":   {"coordinates": [115.8581, 28.6829], "city_tier": 2, "avg_days": 2, "region": "华东",  "tags": ["滕王阁", "八一起义", "鄱阳湖", "瓦罐汤"]},
    "上饶":   {"coordinates": [117.9433, 28.4549], "city_tier": 3, "avg_days": 2, "region": "华东",  "tags": ["三清山", "婺源", "灵山", "茶文化"]},
    "黄山":   {"coordinates": [118.3378, 29.7148], "city_tier": 2, "avg_days": 2, "region": "华东",  "tags": ["奇松怪石", "云海", "温泉", "徽派建筑"]},
    "张家界": {"coordinates": [110.4788, 29.1170], "city_tier": 2, "avg_days": 3, "region": "华中",  "tags": ["悬浮山", "玻璃桥", "土家族", "天门山"]},
    "西双版纳傣族自治州": {"coordinates": [100.7977, 22.0007], "city_tier": 2, "avg_days": 3, "region": "西南", "tags": ["热带雨林", "傣族文化", "泼水节", "东南亚风情"]},
    "丽江":   {"coordinates": [100.2330, 26.8721], "city_tier": 2, "avg_days": 3, "region": "西南",  "tags": ["古城", "雪山", "纳西文化", "浪漫慢城"]},
    "太原":   {"coordinates": [112.5490, 37.8570], "city_tier": 2, "avg_days": 2, "region": "华北",  "tags": ["晋商文化", "平遥古城", "老陈醋", "煤炭资源"]},
    "石家庄": {"coordinates": [114.5149, 38.0428], "city_tier": 2, "avg_days": 1, "region": "华北",  "tags": ["正定古城", "赵州桥", "冀菜", "交通枢纽"]},
    "兰州":   {"coordinates": [103.7940, 36.0611], "city_tier": 2, "avg_days": 2, "region": "西北",  "tags": ["牛肉面", "黄河风情", "西北门户", "白塔山"]},
    "北海":   {"coordinates": [109.1193, 21.4816], "city_tier": 3, "avg_days": 2, "region": "华南",  "tags": ["银滩", "海鲜", "海上丝路", "涠洲岛"]},
    "大同":   {"coordinates": [113.2952, 40.0900], "city_tier": 3, "avg_days": 2, "region": "华北",  "tags": ["云冈石窟", "古城墙", "煤都", "北魏文化"]},
    "银川":   {"coordinates": [106.2309, 38.4872], "city_tier": 2, "avg_days": 2, "region": "西北",  "tags": ["西夏王陵", "沙漠", "回族文化", "葡萄酒"]},
    "西宁":   {"coordinates": [101.7782, 36.6171], "city_tier": 2, "avg_days": 2, "region": "西北",  "tags": ["青藏门户", "塔尔寺", "高原", "清真美食"]},
    "乌鲁木齐": {"coordinates": [87.6168, 43.8256], "city_tier": 2, "avg_days": 3, "region": "西北", "tags": ["新疆", "丝绸之路", "大巴扎", "哈萨克文化"]},
    "呼和浩特": {"coordinates": [111.7520, 40.8415], "city_tier": 2, "avg_days": 2, "region": "华北", "tags": ["草原", "蒙古族文化", "奶茶", "昭君博物馆"]},
    "宜宾":   {"coordinates": [104.6419, 28.7527], "city_tier": 3, "avg_days": 2, "region": "西南",  "tags": ["五粮液", "竹海", "僰道古城", "茶马古道"]},
    "乐山":   {"coordinates": [103.7660, 29.5523], "city_tier": 3, "avg_days": 1, "region": "西南",  "tags": ["乐山大佛", "峨眉山", "佛教文化", "钵钵鸡"]},
    "泉州":   {"coordinates": [118.5893, 24.9139], "city_tier": 2, "avg_days": 2, "region": "华东",  "tags": ["海丝起点", "闽南建筑", "宗教文化", "木偶戏"]},
    "湖州":   {"coordinates": [120.0880, 30.8932], "city_tier": 3, "avg_days": 2, "region": "华东",  "tags": ["南浔古镇", "太湖", "丝绸", "安吉竹海"]},
    "绍兴":   {"coordinates": [120.5800, 30.0300], "city_tier": 3, "avg_days": 2, "region": "华东",  "tags": ["鲁迅故里", "黄酒", "乌篷船", "兰亭"]},
    "金华":   {"coordinates": [119.6479, 29.0788], "city_tier": 3, "avg_days": 1, "region": "华东",  "tags": ["义乌小商品", "横店影视城", "金华火腿", "双龙洞"]},
    "舟山":   {"coordinates": [122.1069, 29.9887], "city_tier": 3, "avg_days": 2, "region": "华东",  "tags": ["普陀山", "海岛", "海鲜", "观音文化"]},
    "台州":   {"coordinates": [121.4280, 28.6565], "city_tier": 3, "avg_days": 2, "region": "华东",  "tags": ["天台山", "神仙居", "海鲜", "宗教文化"]},
    "丽水":   {"coordinates": [119.9221, 28.4676], "city_tier": 3, "avg_days": 2, "region": "华东",  "tags": ["古堰画乡", "云和梯田", "畲族文化", "生态旅游"]},
    "芜湖":   {"coordinates": [118.3762, 31.3336], "city_tier": 3, "avg_days": 1, "region": "华东",  "tags": ["方特主题乐园", "徽派文化", "长江", "芜湖铁画"]},
    "蚌埠":   {"coordinates": [117.3889, 32.9160], "city_tier": 3, "avg_days": 1, "region": "华东",  "tags": ["花鼓灯", "垓下古战场", "龙子湖", "淮河文化"]},
    "景德镇": {"coordinates": [117.2147, 29.2861], "city_tier": 3, "avg_days": 2, "region": "华东",  "tags": ["陶瓷", "千年瓷都", "瓷器制作体验", "古窑"]},
    "赣州":   {"coordinates": [114.9330, 25.8311], "city_tier": 3, "avg_days": 2, "region": "华东",  "tags": ["客家文化", "通天岩", "红色文化", "脐橙"]},
    "柳州":   {"coordinates": [109.3999, 24.3249], "city_tier": 3, "avg_days": 2, "region": "华南",  "tags": ["螺蛳粉", "工业城市", "奇石", "三江程阳风雨桥"]},
    "百色":   {"coordinates": [106.6181, 23.9026], "city_tier": 3, "avg_days": 2, "region": "华南",  "tags": ["红色文化", "靖西通灵大峡谷", "壮族文化", "芒果"]},
    "安顺":   {"coordinates": [105.9320, 26.2453], "city_tier": 3, "avg_days": 2, "region": "西南",  "tags": ["黄果树瀑布", "龙宫", "屯堡文化", "蜡染"]},
    "黔东南苗族侗族自治州（凯里）": {"coordinates": [107.9773, 26.5834], "city_tier": 3, "avg_days": 3, "region": "西南", "tags": ["苗族文化", "侗族鼓楼", "西江千户苗寨", "民族风情"]},
    "拉萨":   {"coordinates": [91.1409, 29.6500],  "city_tier": 2, "avg_days": 4, "region": "西南",  "tags": ["布达拉宫", "藏传佛教", "高原", "雪域圣城"]},
    "嘉兴":   {"coordinates": [120.7512, 30.7522], "city_tier": 3, "avg_days": 1, "region": "华东",  "tags": ["南湖", "红船", "江南水乡", "粽子"]},
    "镇江":   {"coordinates": [119.4552, 32.2044], "city_tier": 3, "avg_days": 1, "region": "华东",  "tags": ["金山寺", "镇江醋", "锅盖面", "扬子江"]},
    "扬州":   {"coordinates": [119.4127, 32.3942], "city_tier": 3, "avg_days": 2, "region": "华东",  "tags": ["瘦西湖", "扬州炒饭", "园林", "古运河"]},
    "连云港": {"coordinates": [119.1788, 34.6003], "city_tier": 3, "avg_days": 2, "region": "华东",  "tags": ["花果山", "海滨", "丝路起点", "海鲜"]},
    "南通":   {"coordinates": [120.8946, 32.0153], "city_tier": 3, "avg_days": 1, "region": "华东",  "tags": ["狼山", "蓝印花布", "江海文化", "纺织城"]},
    "常州":   {"coordinates": [119.9741, 31.7770], "city_tier": 3, "avg_days": 1, "region": "华东",  "tags": ["恐龙园", "天宁寺", "运河古迹", "梳篦"]},
    "无锡":   {"coordinates": [120.3119, 31.4912], "city_tier": 2, "avg_days": 2, "region": "华东",  "tags": ["太湖", "灵山大佛", "无锡菜", "惠山古镇"]},
    "牡丹江": {"coordinates": [129.6328, 44.5521], "city_tier": 3, "avg_days": 2, "region": "东北",  "tags": ["镜泊湖", "雪乡", "朝鲜族文化", "边境旅游"]},
    "齐齐哈尔": {"coordinates": [123.9182, 47.3542], "city_tier": 3, "avg_days": 2, "region": "东北", "tags": ["扎龙湿地", "丹顶鹤", "嫩江", "烤肉"]},
    "延边朝鲜族自治州（延吉）": {"coordinates": [129.5085, 42.9046], "city_tier": 3, "avg_days": 2, "region": "东北", "tags": ["朝鲜族文化", "长白山门户", "冷面", "图们江"]},
    "吉林市": {"coordinates": [126.5500, 43.8380], "city_tier": 3, "avg_days": 2, "region": "东北",  "tags": ["雾凇", "松花湖", "查干湖冬捕", "滑雪"]},
    "锦州":   {"coordinates": [121.1268, 41.0950], "city_tier": 3, "avg_days": 1, "region": "东北",  "tags": ["医巫闾山", "笔架山", "锦州烧烤", "辽西走廊"]},
    "丹东":   {"coordinates": [124.3532, 40.1290], "city_tier": 3, "avg_days": 2, "region": "东北",  "tags": ["鸭绿江", "虎山长城", "朝鲜风情", "黄蚬子"]},
    "抚顺":   {"coordinates": [123.9578, 41.8788], "city_tier": 3, "avg_days": 1, "region": "东北",  "tags": ["煤都", "清永陵", "萨尔浒", "满族文化"]},
    "鞍山":   {"coordinates": [122.9953, 41.1084], "city_tier": 3, "avg_days": 1, "region": "东北",  "tags": ["钢铁城市", "千山", "汤岗子温泉", "玉佛苑"]},
    "呼伦贝尔": {"coordinates": [119.7585, 49.2155], "city_tier": 2, "avg_days": 4, "region": "华北", "tags": ["大草原", "额尔古纳湿地", "俄罗斯风情", "蒙古族文化"]},
    "鄂尔多斯": {"coordinates": [109.7815, 39.6086], "city_tier": 3, "avg_days": 2, "region": "华北", "tags": ["响沙湾", "成吉思汗陵", "草原", "煤炭资源"]},
    "承德":   {"coordinates": [117.9628, 40.9523], "city_tier": 3, "avg_days": 2, "region": "华北",  "tags": ["避暑山庄", "外八庙", "清代皇家园林", "坝上草原"]},
    "张家口": {"coordinates": [114.8840, 40.8244], "city_tier": 3, "avg_days": 2, "region": "华北",  "tags": ["冬奥会", "滑雪", "草原天路", "口蘑"]},
    "秦皇岛": {"coordinates": [119.5997, 39.9354], "city_tier": 3, "avg_days": 2, "region": "华北",  "tags": ["北戴河", "山海关", "长城入海", "海滨避暑"]},
    "香港":   {"coordinates": [114.1694, 22.3193], "city_tier": 1, "avg_days": 3, "region": "华南",  "tags": ["购物天堂", "粤菜", "维多利亚港", "国际金融"]},
}

# 添加项目根目录到路径 (假设脚本在 .claude/skills/ask-question/script/)
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import LLM_CONFIG
from langchain_openai import ChatOpenAI

# 动态加载同目录下的 agent.py
def load_rag_agent_class():
    agent_script = current_dir / "agent.py"
    spec = importlib.util.spec_from_file_location("RAGKnowledgeAgentModule", agent_script)
    module = importlib.util.module_from_spec(spec)
    sys.modules["RAGKnowledgeAgentModule"] = module
    spec.loader.exec_module(module)
    return module.RAGKnowledgeAgent

RAGKnowledgeAgent = load_rag_agent_class()

def split_text(text: str, max_chars: int = 600, overlap: int = 100) -> List[str]:
    """
    简单的文本切分：优先按段落切分，控制每块大小
    """
    chunks = []
    
    # 预处理：按空行分割成段落
    lines = text.split('\n')
    paragraphs = []
    current_para = []
    
    for line in lines:
        if line.strip() == "":
            if current_para:
                paragraphs.append("\n".join(current_para))
                current_para = []
        else:
            current_para.append(line)
    if current_para:
        paragraphs.append("\n".join(current_para))
    
    # 组合段落
    current_chunk = ""
    
    for para in paragraphs:
        # 如果加上当前段落还未超限
        if len(current_chunk) + len(para) <= max_chars:
            current_chunk += "\n\n" + para
        else:
            # 已经超限，先保存当前 chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # 如果单个段落非常长，强制切分
            if len(para) > max_chars:
                # 这里的逻辑简单处理：直接把长段落作为新起点（可能会再次被切分，如果这里加递归太复杂，
                # 简单起见，如果段落超长，就按长度硬切）
                remaining = para
                while len(remaining) > max_chars:
                    chunks.append(remaining[:max_chars])
                    remaining = remaining[max_chars - overlap:]
                current_chunk = remaining
            else:
                # 开启新 chunk，并带上前一个 chunk 的尾部作为 overlap（如果需要）
                # 这里简单起见，不搞 overlap 了，因为是按自然段落切的
                current_chunk = para
    
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

def parse_china_tourist_kb(file_path: str) -> List[Dict]:
    """
    专用解析器：针对 China_Tourist_Knowledge_Base.md 的两级 Markdown 结构
    生成两类 chunk：
      - city_overview：整个城市内容，用于宽泛的城市查询
      - section：带 [城市 > 分类] breadcrumb 的小块，用于精准分类查询
    每个 chunk 的 metadata 中注入 CITY_METADATA 结构化字段，供地图 API 直接提取参数
    """
    documents = []

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 按 ## 城市 切割（保留分隔符所在行）
    city_blocks = re.split(r"\n(?=## )", content)

    for block in city_blocks:
        block = block.strip()
        if not block or block.startswith("# "):
            continue

        first_line = block.split("\n")[0]
        city_name = first_line.lstrip("#").strip()
        if not city_name:
            continue

        city_meta = CITY_METADATA.get(city_name, {})
        base_metadata = {
            "category": "城市旅游",
            "city": city_name,
            "source": "中国旅游知识库",
            "version": "2024版",
            "parent_doc": "China_Tourist_Knowledge_Base.md",
            **city_meta,          # 坐标、等级、天数、region、tags
        }

        # ── city_overview chunk 已删除 ──────────────────────────────────
        # 原因：city_overview 不携带 section 字段，在 city 级过滤时会被高频召回，
        # 导致内容串台（经验 / 避坑 / 住宿全部混入同一结果）。
        # 现在只生成带 section 字段的细粒度块，由调用方通过 section_filter 精准控制。

        # ── Section-level chunks（带 breadcrumb）────────────────────────
        # 按 ### 分割，第一段是城市级 header，跳过
        section_blocks = re.split(r"\n(?=### )", block)
        for sec in section_blocks[1:]:
            sec = sec.strip()
            if not sec:
                continue

            sec_first_line = sec.split("\n")[0]
            section_name = sec_first_line.lstrip("#").strip()
            breadcrumb_content = f"[{city_name} > {section_name}]\n{sec}"

            # 超长 section 继续细分（一般不会超，保留兜底）
            if len(breadcrumb_content) > 800:
                sub_chunks = split_text(breadcrumb_content, max_chars=800, overlap=100)
                for i, sub in enumerate(sub_chunks):
                    documents.append({
                        "id": f"city_section_{city_name}_{section_name}_{i+1}",
                        "content": sub,
                        "metadata": {
                            **base_metadata,
                            "chunk_type": "section",
                            "section": section_name,
                            "title": f"{city_name} - {section_name} (Part {i+1})",
                        },
                    })
            else:
                documents.append({
                    "id": f"city_section_{city_name}_{section_name}",
                    "content": breadcrumb_content,
                    "metadata": {
                        **base_metadata,
                        "chunk_type": "section",
                        "section": section_name,
                        "title": f"{city_name} - {section_name}",
                    },
                })

    return documents


def load_documents_from_directory(directory_path: str) -> List[Dict]:
    """
    从指定目录加载所有文档
    - China_Tourist_Knowledge_Base.md：使用专用两级解析器
    - 其余 .txt 文件：使用通用段落切分器
    """
    documents = []
    doc_dir = Path(directory_path)

    if not doc_dir.exists():
        print(f"[错误] 文档目录不存在: {directory_path}")
        return documents

    # ── 处理 China_Tourist_Knowledge_Base.md ────────────────────────────
    kb_md = doc_dir / "01_china_tourist_knowledge_base.md"
    if kb_md.exists():
        try:
            md_docs = parse_china_tourist_kb(str(kb_md))
            documents.extend(md_docs)
            section_count = len([d for d in md_docs if d["metadata"].get("chunk_type") == "section"])
            print(f"   [成功] 加载文档: {kb_md.name} -> {len(md_docs)} chunks ({section_count} section 块，无 city_overview)")
        except Exception as e:
            print(f"   [错误] 加载 {kb_md.name} 失败: {e}")

    # ── 处理其余 .txt 文件 ───────────────────────────────────────────────
    category_mapping = {
        "booking_guide":         "预订指南",
        "emergency_procedures":  "应急指南",
        "practical_travel_tips": "实用旅行贴士",
    }

    for file_path in sorted(doc_dir.glob("*.txt")):
        try:
            filename_parts = file_path.stem.split("_", 1)
            doc_num = filename_parts[0]
            doc_key = filename_parts[1] if len(filename_parts) > 1 else ""
            base_doc_id = f"doc_{doc_num}"

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                print(f"   [跳过] 空文件: {file_path.name}")
                continue

            title = content.split("\n")[0].strip() or file_path.stem

            category = "旅游知识"
            for key, cat in category_mapping.items():
                if key in doc_key:
                    category = cat
                    break

            chunks = split_text(content, max_chars=600, overlap=100)
            for i, chunk_content in enumerate(chunks):
                documents.append({
                    "id": f"{base_doc_id}_{i+1}",
                    "content": chunk_content,
                    "metadata": {
                        "category": category,
                        "title": f"{title} (Part {i+1})",
                        "source": "旅游知识库文档",
                        "file_path": str(file_path),
                        "version": "2024版",
                        "parent_doc": file_path.name,
                    },
                })

            print(f"   [成功] 加载文档: {file_path.name} -> {len(chunks)} chunks")

        except Exception as e:
            print(f"   [错误] 加载文件失败 {file_path.name}: {e}")
            continue

    return documents


def main():
    print("="*70)
    print("初始化RAG知识库 (Plugin Version) - With Chunking")
    print("="*70)
    print()

    rag_agent = None
    try:
        # 创建模型
        print("1. 创建模型...")
        model = ChatOpenAI(
            model=LLM_CONFIG["model_name"],
            api_key=LLM_CONFIG["api_key"],
            base_url=LLM_CONFIG["base_url"],
            temperature=LLM_CONFIG.get("temperature", 0.7),
            max_tokens=LLM_CONFIG.get("max_tokens", 2000),
        )
        print("[成功] 模型创建成功")
        print()

        # 定义路径
        skill_root = current_dir.parent
        knowledge_base_path = skill_root / "data" / "rag_knowledge"
        documents_dir = skill_root / "data" / "documents"

        # 确保目录存在
        knowledge_base_path.mkdir(parents=True, exist_ok=True)
        
        # 创建RAG Agent
        print("2. 初始化RAG Agent...")
        print(f"   知识库路径: {knowledge_base_path}")
        rag_agent = RAGKnowledgeAgent(
            name="RAGKnowledgeAgent",
            model=model,
            knowledge_base_path=str(knowledge_base_path),
            collection_name="travel_knowledge",
            top_k=5
        )

        if not rag_agent.initialized:
            print("[错误] RAG Agent初始化失败")
            return

        print("[成功] RAG Agent初始化成功")
        print()

        # 从文件加载文档
        print(f"3. 从 {documents_dir} 加载文档...")
        documents = load_documents_from_directory(str(documents_dir))

        if not documents:
            print("[错误] 未加载到任何文档")
            return

        print(f"[成功] 成功切分并加载 {len(documents)} 个片段")
        print()

        # 添加文档到RAG知识库
        print("4. 将文档添加到RAG知识库...")
        
        # 在添加之前，先清空旧的 collection，避免重复索引（本脚本使用 ChromaDB agent）
        try:
            rag_agent.chroma_client.delete_collection(name=rag_agent.collection_name)
            print("   [提示] 检测到已存在 Collection，已删除...")
            rag_agent.collection = rag_agent.chroma_client.get_or_create_collection(
                name=rag_agent.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print("   [成功] Collection 重建完成")
        except Exception as e:
            print(f"   [警告] 清空 Collection 时出错: {e}")

        result = rag_agent.add_documents(documents)

        if result["status"] == "success":
            print(f"[成功] 成功添加 {result['added_count']} 个片段")
            print(f"[成功] 知识库总文档数: {result['total_count']}")
        else:
            print(f"[错误] 添加文档失败: {result.get('message', 'Unknown error')}")
            return

        print()

        # 获取统计信息
        print("5. 知识库统计信息:")
        stats = rag_agent.get_stats()
        if stats["status"] == "success":
            print(f"   - Collection: {stats.get('collection_name')}")
            print(f"   - 文档数量: {stats.get('total_documents')}")
            print(f"   - 存储路径: {stats.get('knowledge_base_path')}")
        print()

        # 测试检索
        print("6. 测试知识检索...")
        test_queries = [
            "上海有哪些必吃美食？",
            "北京最佳旅游季节是什么时候？",
            "旅行前需要做哪些准备？",
        ]

        for query in test_queries:
            print(f"\n   查询: {query}")
            results = rag_agent.search_knowledge(query, top_k=2)
            if results:
                print(f"   [成功] 找到 {len(results)} 个相关文档")
                for i, doc in enumerate(results, 1):
                    # 安全获取 metadata
                    metadata = doc.get('metadata', {})
                    if isinstance(metadata, str):
                        try:
                            import json
                            metadata = json.loads(metadata)
                        except:
                            metadata = {}
                    
                    title = metadata.get('title', 'Unknown')
                    distance = doc.get('distance', 0.0)
                    print(f"      [{i}] {title} (相似度: {1-distance:.3f})")
            else:
                print("   [未找到] 无相关文档")

        print()
        print("="*70)
        print("知识库初始化完成！")
        print("="*70)

    finally:
        # 确保资源被正确清理
        if rag_agent:
            print("\n正在清理资源...")
            try:
                rag_agent.close()
            except:
                pass
            print("[成功] 资源清理完成")


if __name__ == "__main__":
    main()
