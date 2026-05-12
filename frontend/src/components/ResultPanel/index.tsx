import { useEffect, useMemo, useState } from 'react';
import { useChatStore } from '../../store/chatStore';
import EmptyState from './EmptyState';
import TransportTable from './TransportTable';
import DayTabs from './DayTabs';
import DayItineraryTable, { DayRestaurantList } from './DayItineraryTable';
import AmapView from './AmapView';

export default function ResultPanel() {
  const finalData = useChatStore((s) => s.finalData);

  // 早退：非 planning / 没有结构化数据 → 始终保持 EmptyState，不暴露原始 JSON
  if (!finalData || finalData.result_type !== 'planning') {
    return <EmptyState />;
  }

  const transport = finalData.transport_options || [];
  const transportReturn = finalData.transport_return_options || [];
  const dailyRoutes = finalData.daily_routes || [];
  const dailyRestaurants: any[] = finalData.daily_restaurants || [];
  const poiDescriptions = finalData.poi_descriptions || {};
  const poiPhotos = finalData.poi_photos || {};

  const hasAnyStructured =
    transport.length > 0 || transportReturn.length > 0 || dailyRoutes.length > 0;
  if (!hasAnyStructured) {
    return <EmptyState />;
  }

  return (
    <div style={{ padding: 16, overflow: 'auto', height: '100%' }}>
      {transport.length > 0 && (
        <TransportTable options={transport} title="去程交通方案" />
      )}
      {transportReturn.length > 0 && (
        <TransportTable options={transportReturn} title="返程交通方案" />
      )}
      {dailyRoutes.length > 0 && (
        <ItinerarySection
          dailyRoutes={dailyRoutes}
          dailyRestaurants={dailyRestaurants}
          poiDescriptions={poiDescriptions}
          poiPhotos={poiPhotos}
        />
      )}
    </div>
  );
}

// 行程区块独立成子组件，把 selectedDay 状态封装在内；
// 当 dailyRoutes 整体替换（多轮规划）时，effect 会重置 selected 到首日。
function ItinerarySection({
  dailyRoutes,
  dailyRestaurants,
  poiDescriptions,
  poiPhotos,
}: {
  dailyRoutes: any[];
  dailyRestaurants: any[];
  poiDescriptions: Record<string, string>;
  poiPhotos: Record<string, string[]>;
}) {
  // 按 day 排序，避免后端乱序导致 tab 顺序错乱
  const sortedRoutes = useMemo(
    () => [...dailyRoutes].sort((a, b) => (a.day || 0) - (b.day || 0)),
    [dailyRoutes],
  );
  const days = useMemo(() => sortedRoutes.map((r) => r.day as number), [sortedRoutes]);

  const [selectedDay, setSelectedDay] = useState<number>(days[0] ?? 1);

  // 行程整体替换时，把选中日重置到首日（防止旧 selectedDay 在新 days 中不存在）
  useEffect(() => {
    if (!days.includes(selectedDay) && days.length > 0) {
      setSelectedDay(days[0]);
    }
  }, [days, selectedDay]);

  const currentRoute = sortedRoutes.find((r) => r.day === selectedDay);
  const restaurantsForDay =
    dailyRestaurants.find((d) => d.day === selectedDay)?.restaurants || [];

  return (
    <div>
      <div
        style={{
          fontWeight: 700,
          marginBottom: 12,
          fontSize: 17,
          color: '#1f1f1f',
          borderLeft: '4px solid #1677ff',
          paddingLeft: 10,
          lineHeight: '20px',
        }}
      >
        每日行程
      </div>
      <DayTabs days={days} selected={selectedDay} onSelect={setSelectedDay} />
      {currentRoute ? (
        <>
          <DayItineraryTable
            route={currentRoute}
            poiDescriptions={poiDescriptions}
          />
          <AmapView
            route={currentRoute}
            poiDescriptions={poiDescriptions}
            poiPhotos={poiPhotos}
          />
          <DayRestaurantList restaurants={restaurantsForDay} />
        </>
      ) : (
        <div style={{ color: '#999', padding: 24, textAlign: 'center' }}>
          当天暂无行程数据
        </div>
      )}
    </div>
  );
}
