import { Table, Tag } from 'antd';
import type { ColumnsType } from 'antd/es/table';

/**
 * 把后端的耗时字段格式化为 "H 时 M 分"。
 * - "26:19"  → "26 时 19 分"
 * - "2:00"   → "2 时"
 * - "0:45"   → "45 分"
 * - 其他（已是文字 / 空 / null） → 原样返回或 "-"
 */
/**
 * 把时间字段统一格式化为 "HH:MM"，去掉日期前缀。
 * 解析不到时返回原值或 "?"。
 */
function formatTime(v: string | null | undefined): string {
  if (!v) return '?';
  const m = /(\d{2}:\d{2})/.exec(v);
  return m ? m[1] : v;
}

/**
 * 12306 车次号首字母 → 车型中文名。
 * 参考铁路客运标准：
 *   G/C → 高铁  D → 动车  Z → 直达特快  T → 特快  K → 快速
 *   Y → 旅游  L → 临客  纯数字 → 普客
 * 后端 transport_type 仅给"火车"粗类；此函数把车次号细化为车型。
 * 无法识别（缺车次号、未知前缀）时回退到 fallback（即原 transport_type "火车"）。
 */
function trainTypeFromNo(no: string | null | undefined, fallback: string): string {
  if (!no) return fallback;
  const head = no.trim().charAt(0).toUpperCase();
  switch (head) {
    case 'G': return '高铁';
    case 'C': return '高铁';   // 城际高铁，按高铁归口展示
    case 'D': return '动车';
    case 'Z': return '直达';
    case 'T': return '特快';
    case 'K': return '快速';
    case 'Y': return '旅游';
    case 'L': return '临客';
    default:
      // 纯数字开头视作普客
      if (/^\d/.test(head)) return '普客';
      return fallback;
  }
}

function formatDuration(v?: string | null): string {
  if (!v) return '-';
  const m = /^(\d+):(\d+)$/.exec(v.trim());
  if (!m) return v;
  const h = parseInt(m[1], 10);
  const min = parseInt(m[2], 10);
  if (h === 0) return `${min} 分`;
  if (min === 0) return `${h} 时`;
  return `${h} 时 ${min} 分`;
}

export interface TransportOption {
  transport_type?: string;
  transport_no?: string | null;
  departure_time?: string | null;
  arrival_time?: string | null;
  duration?: string;
  departure_hub?: string;
  arrival_hub?: string;
  price_range?: string;
  flight_company?: string | null;
  cabin_class?: string | null;
  is_recommended?: boolean;
  data_source?: string;
  pros?: string;
  cons?: string;
}

interface Props {
  options: TransportOption[];
  title?: string;
}

export default function TransportTable({ options, title = '交通方案' }: Props) {
  if (!options || options.length === 0) return null;

  // 推荐方案排到第一行
  const sorted = [...options].sort(
    (a, b) => Number(b.is_recommended) - Number(a.is_recommended),
  );

  const hasFlight = options.some((o) => o.transport_type === '飞机');

  const columns: ColumnsType<TransportOption> = [
    {
      title: '类型',
      dataIndex: 'transport_type',
      width: 70,
      align: 'center',
      render: (v: string, row) => {
        // 飞机直接用后端值；火车按车次号首字母细化（G/C→高铁、D→动车、Z/T/K…）
        const label = v === '火车' ? trainTypeFromNo(row.transport_no, v) : v;
        return (
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: 2,
            }}
          >
            <span>{label}</span>
            {row.is_recommended && (
              <Tag
                color="gold"
                style={{ margin: 0, fontSize: 11, padding: '0 6px', lineHeight: '18px' }}
              >
                推荐
              </Tag>
            )}
          </div>
        );
      },
    },
    {
      title: '班次',
      width: hasFlight ? 110 : 90,
      align: 'center',
      render: (_, row) => (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', lineHeight: 1.3 }}>
          <span>{row.transport_no || '-'}</span>
          {row.transport_type === '飞机' && row.flight_company && (
            <span style={{ color: '#888', fontSize: 12 }}>{row.flight_company}</span>
          )}
        </div>
      ),
    },
    {
      title: '舱位/席别',
      dataIndex: 'cabin_class',
      width: 90,
      align: 'center',
      // 飞机：后端 normalize_flight 给 cabin_class；火车：_build_train_option 给席别名
      // 兜底为破折号；如需"火车缺值默认二等座"，改成下面这一行即可
      // render: (_, row) => row.cabin_class || (row.transport_type === '飞机' ? '-' : '二等座'),
      render: (v) => v || '-',
    },
    {
      title: '时间',
      width: 130,
      align: 'center',
      render: (_, row) => {
        const dep = formatTime(row.departure_time);
        const arr = formatTime(row.arrival_time);
        return <span style={{ whiteSpace: 'nowrap' }}>{dep} → {arr}</span>;
      },
    },
    {
      title: '耗时',
      dataIndex: 'duration',
      width: 100,
      align: 'center',
      render: (v) => <span style={{ whiteSpace: 'nowrap' }}>{formatDuration(v)}</span>,
    },
    {
      title: '枢纽',
      width: 160,
      align: 'center',
      render: (_, row) => (
        <span style={{ color: '#666' }}>
          {row.departure_hub} → {row.arrival_hub}
        </span>
      ),
    },
    {
      title: '价格',
      dataIndex: 'price_range',
      width: 90,
      align: 'center',
      render: (v) => v || '-',
    },
  ];

  return (
    <div style={{ marginBottom: 20 }}>
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
        {title}
      </div>
      <Table<TransportOption>
        size="small"
        rowKey={(row, idx) => `${row.transport_no || row.transport_type}-${idx}`}
        columns={columns}
        dataSource={sorted}
        pagination={false}
        rowClassName={(row) => (row.is_recommended ? 'transport-row-recommended' : '')}
      />
      <style>
        {`
          .transport-row-recommended > td {
            background-color: #fffbe6 !important;
            font-weight: 500;
          }
        `}
      </style>
    </div>
  );
}
