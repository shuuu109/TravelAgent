import { Table, Tag } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import type { DailyRoute } from '../../types/sse';

interface Restaurant {
  name?: string;
  distance_m?: number;
  amap_rating?: string | number;
}

interface DailyRestaurantsItem {
  day: number;
  restaurants?: Restaurant[];
}

interface Props {
  route: DailyRoute;
  restaurantsForDay?: Restaurant[];
  poiDescriptions?: Record<string, string>;
}

// 单天的行程表格 + 周边餐饮区块。地图（阶段 4）会复用 route.ordered_pois。
export default function DayItineraryTable({ route, restaurantsForDay, poiDescriptions }: Props) {
  const pois = route.ordered_pois || [];
  const legs = route.legs || [];
  const descs = poiDescriptions || {};

  // 把 leg 折叠到对应起点 POI 行（leg[i] 表示 pois[i] -> pois[i+1]）
  const rows = pois.map((poi, idx) => ({
    key: `${idx}-${poi.name}`,
    seq: idx + 1,
    poi,
    leg: legs[idx],
    desc: descs[poi.name || ''] || '',
  }));

  type Row = (typeof rows)[number];

  const columns: ColumnsType<Row> = [
    {
      title: '#',
      dataIndex: 'seq',
      width: 40,
      align: 'center',
    },
    {
      title: '景点',
      width: 160,
      render: (_, row) => (
        <div>
          <div style={{ fontWeight: 500 }}>{row.poi.name}</div>
          {row.poi.category && (
            <Tag color="blue" style={{ marginTop: 4, fontSize: 11 }}>
              {row.poi.category as string}
            </Tag>
          )}
        </div>
      ),
    },
    {
      title: '停留',
      width: 70,
      align: 'center',
      render: (_, row) => {
        const h = (row.poi as any).estimated_hours;
        return h ? `${h}h` : '-';
      },
    },
    {
      title: '下一程交通',
      width: 110,
      align: 'center',
      render: (_, row) => {
        if (!row.leg) return <span style={{ color: '#bbb' }}>—</span>;
        const dur = formatDuration(row.leg.duration);
        // 后端 mode 多为 transit/walking 等英文标识，对用户没意义；
        // 仅展示耗时，无耗时数据时退化到 "—"
        if (!dur) return <span style={{ color: '#bbb' }}>—</span>;
        return <span style={{ fontSize: 12, color: '#555' }}>{dur}</span>;
      },
    },
    {
      title: '体验描述',
      render: (_, row) =>
        row.desc ? (
          <span style={{ fontSize: 12, color: '#555', lineHeight: 1.6 }}>{row.desc}</span>
        ) : (
          <span style={{ color: '#bbb' }}>—</span>
        ),
    },
  ];

  const totalDuration = formatDuration(route.total_duration);
  const restaurants = restaurantsForDay || [];

  return (
    <div>
      <Table<Row>
        size="small"
        columns={columns}
        dataSource={rows}
        pagination={false}
      />

      {totalDuration && (
        <div style={metaStyle}>
          总通勤时长：<strong>{totalDuration}</strong>
        </div>
      )}

      {restaurants.length > 0 && (
        <div style={{ marginTop: 12 }}>
          <div style={{ fontWeight: 600, marginBottom: 6, fontSize: 14 }}>
            周边餐饮推荐
          </div>
          <ul style={restaurantListStyle}>
            {restaurants.map((r, i) => (
              <li key={`${i}-${r.name}`} style={restaurantItemStyle}>
                <span style={{ fontWeight: 500 }}>{r.name}</span>
                <span style={{ color: '#999', fontSize: 12, marginLeft: 8 }}>
                  {restaurantMeta(r)}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

// 与后端 _format_duration 对齐：分钟 → "X小时Y分钟" / "Y分钟"
export function formatDuration(minutes?: number): string {
  if (!minutes || minutes <= 0) return '';
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  if (h > 0 && m > 0) return `${h}小时${m}分钟`;
  if (h > 0) return `${h}小时`;
  return `${m}分钟`;
}

function restaurantMeta(r: Restaurant): string {
  const parts: string[] = [];
  if (r.distance_m) parts.push(`约 ${r.distance_m}m`);
  if (r.amap_rating) parts.push(`评分 ${r.amap_rating}`);
  return parts.length ? `· ${parts.join(' · ')}` : '';
}

const metaStyle: React.CSSProperties = {
  marginTop: 8,
  fontSize: 12,
  color: '#666',
};

const restaurantListStyle: React.CSSProperties = {
  margin: 0,
  padding: 0,
  listStyle: 'none',
};

const restaurantItemStyle: React.CSSProperties = {
  padding: '4px 0',
  fontSize: 13,
  borderBottom: '1px dashed #f0f0f0',
};

// re-export to keep external imports stable when callers want to format on their own
export type { DailyRestaurantsItem };
