import { Table, Tag } from 'antd';
import type { ColumnsType } from 'antd/es/table';

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
      render: (v: string, row) => (
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: 2,
          }}
        >
          <span>{v}</span>
          {row.is_recommended && (
            <Tag
              color="gold"
              style={{ margin: 0, fontSize: 11, padding: '0 6px', lineHeight: '18px' }}
            >
              推荐
            </Tag>
          )}
        </div>
      ),
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
        const dep = row.departure_time || '?';
        const arr = row.arrival_time || '?';
        return <span style={{ whiteSpace: 'nowrap' }}>{dep} → {arr}</span>;
      },
    },
    {
      title: '耗时',
      dataIndex: 'duration',
      width: 80,
      align: 'center',
      render: (v) => <span style={{ whiteSpace: 'nowrap' }}>{v || '-'}</span>,
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
      align: 'left',
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
