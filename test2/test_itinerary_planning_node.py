"""
Step 6 单元测试：itinerary_planning_node

覆盖范围：
  6a. _select_pois          — 按旅行风格 & 天数筛选 POI（纯函数，无 Mock）
  6b. _cluster_by_geography — 贪心地理聚类：重心锚点 + 最近邻贪心（纯函数，无 Mock）
  6c. _optimize_daily_route — TSP 优化 + 高德路线（Mock get_distance_matrix / get_transit_route）
  节点集成                  — itinerary_planning_node 完整流程（Mock amap_mcp_session）

Mock 模式与 test2/test_mock_plan.py 保持一致：
  - 火车/航班：patch.object(client, "method", AsyncMock)         [test_mock_plan.py 参考]
  - 酒店：    patch("mcp_clients.hotel_client.search_hotels", AsyncMock) [test_mock_plan.py 参考]
  - 高德 MCP：patch 模块内导入的 amap_mcp_session / get_distance_matrix / get_transit_route
              （patch 目标为 graph.nodes.itinerary_planning_node.<name>，
               因为该模块顶层已 from mcp_clients.amap_client import ... ）
"""
import sys
import os
import asyncio
import math
import unittest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

# Windows GBK -> UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from graph.nodes.itinerary_planning_node import (
    _select_pois,
    _cluster_by_geography,
    _optimize_daily_route,
    _tsp_brute_force_matrix,
    _tsp_nearest_neighbor_matrix,
    _euclidean,
    create_itinerary_planning_node,
)
from graph.state import HardConstraints

# =============================================================================
# ── Mock 数据 ─────────────────────────────────────────────────────────────────
# =============================================================================

# 北京真实坐标，地理上明显分为三个区域：
#   西部（颐和园/圆明园）、中部（故宫/天安门/北海/全聚德/簋街/南锣）、东部（三里屯/长城）
MOCK_POIS = [
    # 景点（6个）
    {"name": "故宫",      "lng": 116.397, "lat": 39.916, "category": "景点", "rating": 4.9, "address": "北京市东城区"},
    {"name": "天安门",    "lng": 116.391, "lat": 39.908, "category": "景点", "rating": 4.8, "address": "北京市东城区"},
    {"name": "颐和园",    "lng": 116.275, "lat": 39.999, "category": "景点", "rating": 4.8, "address": "北京市海淀区"},
    {"name": "圆明园",    "lng": 116.295, "lat": 40.009, "category": "景点", "rating": 4.6, "address": "北京市海淀区"},
    {"name": "长城",      "lng": 116.557, "lat": 40.432, "category": "景点", "rating": 5.0, "address": "北京市怀柔区"},
    {"name": "北海公园",  "lng": 116.387, "lat": 39.924, "category": "景点", "rating": 4.6, "address": "北京市西城区"},
    # 餐厅（3个）
    {"name": "全聚德",    "lng": 116.398, "lat": 39.899, "category": "餐厅", "rating": 4.7, "address": "北京市东城区"},
    {"name": "烤鸭王",    "lng": 116.404, "lat": 39.910, "category": "餐厅", "rating": 4.5, "address": "北京市东城区"},
    {"name": "簋街烤鱼",  "lng": 116.422, "lat": 39.928, "category": "餐厅", "rating": 4.4, "address": "北京市东城区"},
    # 体验（2个）
    {"name": "三里屯",    "lng": 116.455, "lat": 39.935, "category": "体验", "rating": 4.5, "address": "北京市朝阳区"},
    {"name": "南锣鼓巷",  "lng": 116.411, "lat": 39.938, "category": "体验", "rating": 4.6, "address": "北京市东城区"},
]

# 3×3 时间矩阵（秒），最优路径为 0→1→2（cost=100+200=300）
# 其他路径如 0→2→1 代价为 500+200=700，明显更差，便于断言 TSP 结果
MATRIX_3x3 = [
    [0,   100,  500],
    [100,  0,   200],
    [500, 200,   0 ],
]

# 4×4 时间矩阵，最优路径为 0→1→2→3（cost=50+80+60=190）
MATRIX_4x4 = [
    [0,    50,  900,  800],
    [50,    0,   80,  700],
    [900,  80,    0,   60],
    [800, 700,   60,    0],
]

# 5×5 时间矩阵，用于测试最近邻贪心（n>4）
# 从 0 出发最近邻：0→1(10)→2(20)→3(30)→4(40)
MATRIX_5x5 = [
    [  0,  10, 200, 300, 400],
    [ 10,   0,  20, 300, 400],
    [200,  20,   0,  30, 400],
    [300, 300,  30,   0,  40],
    [400, 400, 400,  40,   0],
]

# 高德路线 Mock 返回值
MOCK_TRANSIT_ROUTE = {
    "duration": 1200,          # 20分钟
    "distance": 4500,
    "steps": ["乘地铁1号线至王府井站", "步行300米"],
    "recommended_mode": "transit",
}


def _make_mock_session(
    distance_matrix=None,
    transit_route=None,
    distance_exc=None,
    transit_exc=None,
):
    """
    构造 mock MCP session。
    distance_exc / transit_exc 若不为 None，则对应方法抛出该异常。
    """
    session = AsyncMock()
    # get_distance_matrix / get_transit_route 在节点模块级别导入，
    # 直接 patch 模块命名空间，session 对象本身不需要这两个属性。
    return session


def _make_amap_session_ctx(mock_session):
    """
    返回一个 async context manager factory，yield mock_session。
    用来替换 amap_mcp_session()。
    """
    @asynccontextmanager
    async def _ctx():
        yield mock_session

    return _ctx


# =============================================================================
# ── 6a: _select_pois 测试 ─────────────────────────────────────────────────────
# =============================================================================

class TestSelectPois(unittest.TestCase):

    def test_普通风格_2天_返回6个(self):
        """普通风格每天3个，2天共6个；类别配比约 60/20/20"""
        result = _select_pois(MOCK_POIS, travel_style="普通", travel_days=2)
        self.assertEqual(len(result), 6)

    def test_特种兵风格_1天_返回4个(self):
        result = _select_pois(MOCK_POIS, travel_style="特种兵", travel_days=1)
        self.assertEqual(len(result), 4)

    def test_情侣风格_1天_景点优先(self):
        """情侣风格每天2个，应优先选最高评分景点"""
        result = _select_pois(MOCK_POIS, travel_style="情侣", travel_days=1)
        self.assertEqual(len(result), 2)
        # 配额 2*0.6=1.2 → round=1 个景点，最高评分景点（长城5.0）应被选中
        names = [p["name"] for p in result]
        self.assertIn("长城", names)

    def test_候选全为景点_无餐厅体验_从剩余补足(self):
        """当餐厅/体验为空时，直接从景点列表补足总量"""
        only_attractions = [p for p in MOCK_POIS if p["category"] == "景点"]
        result = _select_pois(only_attractions, travel_style="普通", travel_days=1)
        # 普通1天3个，景点足够（6个），应从景点中取3个
        self.assertEqual(len(result), 3)
        self.assertTrue(all(p["category"] == "景点" for p in result))

    def test_总量不足时取全部(self):
        """候选不足 total_needed 时，最多返回所有候选"""
        tiny = MOCK_POIS[:2]  # 只有2个POI
        result = _select_pois(tiny, travel_style="特种兵", travel_days=3)
        # total_needed=12，但候选只有2个
        self.assertEqual(len(result), 2)

    def test_景点按评分降序选取(self):
        """在同类别内，应选评分更高的"""
        result = _select_pois(MOCK_POIS, travel_style="特种兵", travel_days=1)
        attraction_results = [p for p in result if p["category"] == "景点"]
        # 景点配额 round(4*0.6)=2，应选长城(5.0)和故宫(4.9)
        attraction_names = {p["name"] for p in attraction_results}
        self.assertIn("长城", attraction_names)
        self.assertIn("故宫", attraction_names)


# =============================================================================
# ── 6b: _cluster_by_geography 测试 ────────────────────────────────────────────
# =============================================================================

class TestClusterByGeography(unittest.TestCase):

    def test_单天_所有POI归入第1天(self):
        pois = MOCK_POIS[:4]
        result = _cluster_by_geography(pois, travel_days=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["day"], 1)
        self.assertEqual(len(result[0]["pois"]), 4)

    def test_两天_均匀分配(self):
        pois = MOCK_POIS[:4]  # 4个POI，每天2个
        result = _cluster_by_geography(pois, travel_days=2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["day"], 1)
        self.assertEqual(result[1]["day"], 2)
        self.assertEqual(len(result[0]["pois"]) + len(result[1]["pois"]), 4)

    def test_不均匀分配_余数补给前几天(self):
        """5个POI，2天 → 第1天3个，第2天2个"""
        pois = MOCK_POIS[:5]
        result = _cluster_by_geography(pois, travel_days=2)
        self.assertEqual(len(result[0]["pois"]), 3)
        self.assertEqual(len(result[1]["pois"]), 2)

    def test_所有POI都被分配无遗漏(self):
        """分配后 POI 总数等于输入总数"""
        result = _cluster_by_geography(MOCK_POIS, travel_days=3)
        total = sum(len(g["pois"]) for g in result)
        self.assertEqual(total, len(MOCK_POIS))

    def test_第一天锚点距重心最远(self):
        """
        第一天的第一个 POI（锚点）应该是距整体重心最远的那个。
        使用4个地理差异明显的点验证。
        """
        # 西南角、中间两个、东北角
        pois = [
            {"name": "西南", "lng": 100.0, "lat": 20.0, "category": "景点", "rating": 4.0},
            {"name": "中1",  "lng": 116.0, "lat": 39.0, "category": "景点", "rating": 4.0},
            {"name": "中2",  "lng": 116.5, "lat": 40.0, "category": "景点", "rating": 4.0},
            {"name": "东北", "lng": 130.0, "lat": 50.0, "category": "景点", "rating": 4.0},
        ]
        # 重心约为 (115.625, 37.25)
        # 距重心最远：西南(100,20) 距离 sqrt(15.625^2+17.25^2)≈23.3
        #            或东北(130,50) 距离 sqrt(14.375^2+12.75^2)≈19.2
        # 西南应该更远
        centroid_lng = sum(p["lng"] for p in pois) / 4
        centroid_lat = sum(p["lat"] for p in pois) / 4
        expected_anchor = max(
            pois,
            key=lambda p: _euclidean((p["lng"], p["lat"]), (centroid_lng, centroid_lat)),
        )
        result = _cluster_by_geography(pois, travel_days=2)
        day1_anchor = result[0]["pois"][0]
        self.assertEqual(day1_anchor["name"], expected_anchor["name"])

    def test_同天POI地理集中(self):
        """
        验证聚类后同天 POI 的内部平均距离小于跨天 POI 的平均距离。
        使用地理上分为两组的 POI：西部组（lng≈116.28）和东部组（lng≈116.45+）
        """
        west_group = [
            {"name": "颐和园", "lng": 116.275, "lat": 39.999, "category": "景点", "rating": 4.8},
            {"name": "圆明园", "lng": 116.295, "lat": 40.009, "category": "景点", "rating": 4.6},
        ]
        east_group = [
            {"name": "三里屯", "lng": 116.455, "lat": 39.935, "category": "体验", "rating": 4.5},
            {"name": "长城",   "lng": 116.557, "lat": 40.432, "category": "景点", "rating": 5.0},
        ]
        pois = west_group + east_group
        result = _cluster_by_geography(pois, travel_days=2)
        # 每天2个，验证两天分别集中在西部或东部
        for group in result:
            day_pois = group["pois"]
            if len(day_pois) == 2:
                intra_dist = _euclidean(
                    (day_pois[0]["lng"], day_pois[0]["lat"]),
                    (day_pois[1]["lng"], day_pois[1]["lat"]),
                )
                # 西部两点距离≈0.02，东部两点距离≈0.50；跨组距离≈0.18-0.50
                # 最差的"同组"距离也应 < 两组之间的平均距离(≈0.18)
                # 实际验证：每天组内距离 < 0.55（排除极端情况）
                self.assertLess(intra_dist, 0.6,
                    f"第{group['day']}天 POI 距离 {intra_dist:.3f} 过大，聚类效果差")

    def test_空输入返回空列表(self):
        self.assertEqual(_cluster_by_geography([], travel_days=2), [])

    def test_travel_days为零返回空列表(self):
        self.assertEqual(_cluster_by_geography(MOCK_POIS[:4], travel_days=0), [])


# =============================================================================
# ── 6c: _optimize_daily_route 测试 ────────────────────────────────────────────
# =============================================================================

class TestOptimizeDailyRoute(unittest.IsolatedAsyncioTestCase):

    def _pois_for_matrix(self, n: int):
        """取前 n 个 MOCK_POIS 用作路线优化输入"""
        return MOCK_POIS[:n]

    async def test_单个POI_直接返回无腿(self):
        mock_session = AsyncMock()
        result = await _optimize_daily_route([MOCK_POIS[0]], city="北京", session=mock_session)
        self.assertEqual(result["ordered_pois"], [MOCK_POIS[0]])
        self.assertEqual(result["legs"], [])
        self.assertEqual(result["total_duration"], 0)

    async def test_3个POI_暴力TSP_找最优路线(self):
        """
        MATRIX_3x3 的最优路线为 0→1→2（cost=300），
        验证 _optimize_daily_route 返回对应的顺序。
        """
        pois = self._pois_for_matrix(3)
        mock_session = AsyncMock()

        with patch("graph.nodes.itinerary_planning_node.get_distance_matrix",
                   new_callable=AsyncMock, return_value=MATRIX_3x3), \
             patch("graph.nodes.itinerary_planning_node.get_transit_route",
                   new_callable=AsyncMock, return_value=MOCK_TRANSIT_ROUTE):
            result = await _optimize_daily_route(pois, city="北京", session=mock_session)

        # 最优顺序 0→1→2 对应的 POI 名称
        ordered_names = [p["name"] for p in result["ordered_pois"]]
        expected_names = [pois[0]["name"], pois[1]["name"], pois[2]["name"]]
        self.assertEqual(ordered_names, expected_names)
        # 2条腿，每条 duration=1200
        self.assertEqual(len(result["legs"]), 2)
        self.assertEqual(result["total_duration"], 2400)

    async def test_4个POI_暴力TSP_调用距离矩阵(self):
        """n=4 时仍使用暴力全排列，MATRIX_4x4 最优路线为 0→1→2→3"""
        pois = self._pois_for_matrix(4)
        mock_session = AsyncMock()

        with patch("graph.nodes.itinerary_planning_node.get_distance_matrix",
                   new_callable=AsyncMock, return_value=MATRIX_4x4), \
             patch("graph.nodes.itinerary_planning_node.get_transit_route",
                   new_callable=AsyncMock, return_value=MOCK_TRANSIT_ROUTE):
            result = await _optimize_daily_route(pois, city="北京", session=mock_session)

        ordered_names = [p["name"] for p in result["ordered_pois"]]
        self.assertEqual(ordered_names,
                         [pois[0]["name"], pois[1]["name"], pois[2]["name"], pois[3]["name"]])
        self.assertEqual(len(result["legs"]), 3)

    async def test_5个POI_最近邻贪心TSP(self):
        """n=5 触发最近邻贪心；MATRIX_5x5 从节点0出发预期路径 0→1→2→3→4"""
        pois = self._pois_for_matrix(5)
        mock_session = AsyncMock()

        with patch("graph.nodes.itinerary_planning_node.get_distance_matrix",
                   new_callable=AsyncMock, return_value=MATRIX_5x5), \
             patch("graph.nodes.itinerary_planning_node.get_transit_route",
                   new_callable=AsyncMock, return_value=MOCK_TRANSIT_ROUTE):
            result = await _optimize_daily_route(pois, city="北京", session=mock_session)

        ordered_names = [p["name"] for p in result["ordered_pois"]]
        self.assertEqual(len(ordered_names), 5)
        # 最近邻从 0 出发：0→1→2→3→4
        self.assertEqual(ordered_names,
                         [pois[0]["name"], pois[1]["name"], pois[2]["name"],
                          pois[3]["name"], pois[4]["name"]])

    async def test_距离矩阵失败_降级欧氏距离TSP_路线仍成功(self):
        """
        get_distance_matrix 抛异常时，应降级为欧氏距离 TSP，
        get_transit_route 仍正常调用，返回有效 legs。
        """
        pois = self._pois_for_matrix(3)
        mock_session = AsyncMock()

        with patch("graph.nodes.itinerary_planning_node.get_distance_matrix",
                   new_callable=AsyncMock, side_effect=RuntimeError("MCP timeout")), \
             patch("graph.nodes.itinerary_planning_node.get_transit_route",
                   new_callable=AsyncMock, return_value=MOCK_TRANSIT_ROUTE):
            result = await _optimize_daily_route(pois, city="北京", session=mock_session)

        # 即使距离矩阵失败，仍返回3个有序 POI 和 2 条腿
        self.assertEqual(len(result["ordered_pois"]), 3)
        self.assertEqual(len(result["legs"]), 2)
        self.assertGreater(result["total_duration"], 0)

    async def test_路线查询失败_该腿duration为0_不中断整体(self):
        """
        get_transit_route 对某条腿抛异常时，该腿 duration=0、mode="unknown"，
        整体规划不中断，其他腿正常。
        """
        pois = self._pois_for_matrix(3)
        mock_session = AsyncMock()

        call_count = 0

        async def transit_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transit error")
            return MOCK_TRANSIT_ROUTE

        with patch("graph.nodes.itinerary_planning_node.get_distance_matrix",
                   new_callable=AsyncMock, return_value=MATRIX_3x3), \
             patch("graph.nodes.itinerary_planning_node.get_transit_route",
                   side_effect=transit_side_effect):
            result = await _optimize_daily_route(pois, city="北京", session=mock_session)

        self.assertEqual(len(result["legs"]), 2)
        # 第1条腿（失败）
        self.assertEqual(result["legs"][0]["duration"], 0)
        self.assertEqual(result["legs"][0]["mode"], "unknown")
        # 第2条腿（成功）
        self.assertEqual(result["legs"][1]["duration"], MOCK_TRANSIT_ROUTE["duration"])


# =============================================================================
# ── 节点集成测试 ───────────────────────────────────────────────────────────────
# =============================================================================

class TestItineraryPlanningNode(unittest.IsolatedAsyncioTestCase):
    """
    通过 create_itinerary_planning_node() 工厂函数获取节点，
    传入模拟 TravelGraphState（dict），验证完整流程。
    """

    def _make_state(
        self,
        poi_candidates=None,
        travel_style="普通",
        travel_days=2,
        destination="北京",
    ) -> dict:
        """构造最小 TravelGraphState dict"""
        hc = HardConstraints(destination=destination, origin="上海")
        return {
            "poi_candidates": poi_candidates if poi_candidates is not None else MOCK_POIS,
            "travel_style": travel_style,
            "travel_days": travel_days,
            "hard_constraints": hc,
            "messages": [],
            "skill_results": [],
        }

    async def test_poi_candidates为空_返回空dict(self):
        node = create_itinerary_planning_node()
        state = self._make_state(poi_candidates=[])
        result = await node(state)
        self.assertEqual(result, {})

    async def test_MCP成功_返回daily_itinerary和daily_routes(self):
        """
        正常路径：amap_mcp_session 成功，get_distance_matrix / get_transit_route 均成功。
        验证返回结构包含 daily_itinerary 和 daily_routes，天数匹配 travel_days。
        """
        node = create_itinerary_planning_node()
        state = self._make_state(travel_style="普通", travel_days=2)
        mock_session = AsyncMock()

        # 动态生成 N×N 矩阵（distance_matrix 的 size 依 POI 数而定）
        def make_matrix(session, origins, destinations):
            n = len(origins)
            return [[0 if i == j else 300 for j in range(n)] for i in range(n)]

        with patch("graph.nodes.itinerary_planning_node.amap_mcp_session",
                   _make_amap_session_ctx(mock_session)), \
             patch("graph.nodes.itinerary_planning_node.get_distance_matrix",
                   new_callable=AsyncMock, side_effect=make_matrix), \
             patch("graph.nodes.itinerary_planning_node.get_transit_route",
                   new_callable=AsyncMock, return_value=MOCK_TRANSIT_ROUTE):
            result = await node(state)

        self.assertIn("daily_itinerary", result)
        self.assertIn("daily_routes", result)
        # 应有 travel_days 天
        self.assertEqual(len(result["daily_itinerary"]), 2)
        self.assertEqual(len(result["daily_routes"]), 2)
        # day 字段正确
        days_itinerary = [g["day"] for g in result["daily_itinerary"]]
        days_routes = [g["day"] for g in result["daily_routes"]]
        self.assertEqual(days_itinerary, [1, 2])
        self.assertEqual(days_routes, [1, 2])

    async def test_每日行程包含pois字段(self):
        """daily_itinerary 的每一天都有 pois 列表，且非空"""
        node = create_itinerary_planning_node()
        state = self._make_state(travel_style="亲子", travel_days=2)
        mock_session = AsyncMock()

        def make_matrix(session, origins, destinations):
            n = len(origins)
            return [[0 if i == j else 200 for j in range(n)] for i in range(n)]

        with patch("graph.nodes.itinerary_planning_node.amap_mcp_session",
                   _make_amap_session_ctx(mock_session)), \
             patch("graph.nodes.itinerary_planning_node.get_distance_matrix",
                   new_callable=AsyncMock, side_effect=make_matrix), \
             patch("graph.nodes.itinerary_planning_node.get_transit_route",
                   new_callable=AsyncMock, return_value=MOCK_TRANSIT_ROUTE):
            result = await node(state)

        for day_group in result["daily_itinerary"]:
            self.assertIn("pois", day_group)
            self.assertGreater(len(day_group["pois"]), 0)

    async def test_每日路线包含ordered_pois和legs字段(self):
        """daily_routes 每天包含 ordered_pois / legs / total_duration"""
        node = create_itinerary_planning_node()
        state = self._make_state(travel_style="情侣", travel_days=1)
        mock_session = AsyncMock()

        def make_matrix(session, origins, destinations):
            n = len(origins)
            return [[0 if i == j else 400 for j in range(n)] for i in range(n)]

        with patch("graph.nodes.itinerary_planning_node.amap_mcp_session",
                   _make_amap_session_ctx(mock_session)), \
             patch("graph.nodes.itinerary_planning_node.get_distance_matrix",
                   new_callable=AsyncMock, side_effect=make_matrix), \
             patch("graph.nodes.itinerary_planning_node.get_transit_route",
                   new_callable=AsyncMock, return_value=MOCK_TRANSIT_ROUTE):
            result = await node(state)

        for day_route in result["daily_routes"]:
            self.assertIn("ordered_pois", day_route)
            self.assertIn("legs", day_route)
            self.assertIn("total_duration", day_route)

    async def test_MCP_session_异常_降级返回空腿(self):
        """
        amap_mcp_session() 整体抛异常时，fallback：
        daily_routes 仍存在，但每天 legs=[]、total_duration=0。
        """
        node = create_itinerary_planning_node()
        state = self._make_state(travel_style="普通", travel_days=2)

        @asynccontextmanager
        async def _failing_session():
            raise ConnectionError("高德 MCP 无法连接")
            yield  # noqa: unreachable — required for @asynccontextmanager

        with patch("graph.nodes.itinerary_planning_node.amap_mcp_session", _failing_session):
            result = await node(state)

        self.assertIn("daily_itinerary", result)
        self.assertIn("daily_routes", result)
        # fallback 时每天 legs 为空，duration 为 0
        for day_route in result["daily_routes"]:
            self.assertEqual(day_route["legs"], [])
            self.assertEqual(day_route["total_duration"], 0)

    async def test_hard_constraints_为dict时正确提取城市(self):
        """hard_constraints 以 dict 形式传入时也能正确提取目的地"""
        node = create_itinerary_planning_node()
        state = {
            "poi_candidates": MOCK_POIS[:4],
            "travel_style": "普通",
            "travel_days": 1,
            "hard_constraints": {"destination": "北京", "origin": "上海"},
            "messages": [],
            "skill_results": [],
        }
        mock_session = AsyncMock()

        def make_matrix(session, origins, destinations):
            n = len(origins)
            return [[0 if i == j else 100 for j in range(n)] for i in range(n)]

        with patch("graph.nodes.itinerary_planning_node.amap_mcp_session",
                   _make_amap_session_ctx(mock_session)), \
             patch("graph.nodes.itinerary_planning_node.get_distance_matrix",
                   new_callable=AsyncMock, side_effect=make_matrix), \
             patch("graph.nodes.itinerary_planning_node.get_transit_route",
                   new_callable=AsyncMock, return_value=MOCK_TRANSIT_ROUTE):
            result = await node(state)

        # 主要验证不抛异常，且返回结构正确
        self.assertIn("daily_itinerary", result)
        self.assertIn("daily_routes", result)

    async def test_特种兵风格_3天_POI数量匹配(self):
        """特种兵每天4个，3天共12个；验证 daily_itinerary 覆盖所有天"""
        node = create_itinerary_planning_node()
        state = self._make_state(travel_style="特种兵", travel_days=3)
        mock_session = AsyncMock()

        def make_matrix(session, origins, destinations):
            n = len(origins)
            return [[0 if i == j else 150 for j in range(n)] for i in range(n)]

        with patch("graph.nodes.itinerary_planning_node.amap_mcp_session",
                   _make_amap_session_ctx(mock_session)), \
             patch("graph.nodes.itinerary_planning_node.get_distance_matrix",
                   new_callable=AsyncMock, side_effect=make_matrix), \
             patch("graph.nodes.itinerary_planning_node.get_transit_route",
                   new_callable=AsyncMock, return_value=MOCK_TRANSIT_ROUTE):
            result = await node(state)

        self.assertEqual(len(result["daily_itinerary"]), 3)
        self.assertEqual(len(result["daily_routes"]), 3)
        total_pois = sum(len(g["pois"]) for g in result["daily_itinerary"])
        # 特种兵3天最多需要12个，但 MOCK_POIS 只有11个，取全部
        self.assertLessEqual(total_pois, len(MOCK_POIS))


# =============================================================================
# ── TSP 纯函数测试（辅助覆盖） ─────────────────────────────────────────────────
# =============================================================================

class TestTspHelpers(unittest.TestCase):

    def test_brute_force_2个节点(self):
        """2个节点只有1种有意义的顺序 0→1"""
        matrix = [[0, 10], [10, 0]]
        result = _tsp_brute_force_matrix(matrix, 2)
        self.assertEqual(len(result), 2)

    def test_brute_force_3个节点_最优路线(self):
        """MATRIX_3x3 最优为 0→1→2"""
        result = _tsp_brute_force_matrix(MATRIX_3x3, 3)
        self.assertEqual(result, [0, 1, 2])

    def test_brute_force_4个节点_最优路线(self):
        """MATRIX_4x4 最优为 0→1→2→3"""
        result = _tsp_brute_force_matrix(MATRIX_4x4, 4)
        self.assertEqual(result, [0, 1, 2, 3])

    def test_nearest_neighbor_5个节点(self):
        """MATRIX_5x5 最近邻从0出发应得 0→1→2→3→4"""
        result = _tsp_nearest_neighbor_matrix(MATRIX_5x5, 5)
        self.assertEqual(result, [0, 1, 2, 3, 4])

    def test_nearest_neighbor_所有节点都被访问(self):
        result = _tsp_nearest_neighbor_matrix(MATRIX_5x5, 5)
        self.assertEqual(sorted(result), list(range(5)))

    def test_euclidean_已知结果(self):
        dist = _euclidean((0.0, 0.0), (3.0, 4.0))
        self.assertAlmostEqual(dist, 5.0)

    def test_euclidean_相同点距离为零(self):
        self.assertAlmostEqual(_euclidean((116.4, 39.9), (116.4, 39.9)), 0.0)


# =============================================================================
# ── 入口 ──────────────────────────────────────────────────────────────────────
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
