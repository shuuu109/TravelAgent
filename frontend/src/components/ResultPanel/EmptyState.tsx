import { Empty } from 'antd';

export default function EmptyState() {
  return (
    <div
      style={{
        height: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <Empty description="规划完成后将在这里显示每日行程与地图" />
    </div>
  );
}
