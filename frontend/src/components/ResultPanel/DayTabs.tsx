interface Props {
  days: number[];
  selected: number;
  onSelect: (day: number) => void;
}

// 横向 tab 栏；selected 由父组件控制（单一选中态）。
export default function DayTabs({ days, selected, onSelect }: Props) {
  return (
    <div style={containerStyle}>
      {days.map((d) => {
        const active = d === selected;
        return (
          <button
            key={d}
            type="button"
            onClick={() => onSelect(d)}
            style={tabStyle(active)}
          >
            Day {d}
          </button>
        );
      })}
    </div>
  );
}

const containerStyle: React.CSSProperties = {
  display: 'flex',
  gap: 8,
  marginBottom: 12,
  borderBottom: '1px solid #e6e8eb',
  paddingBottom: 8,
};

function tabStyle(active: boolean): React.CSSProperties {
  return {
    padding: '4px 12px',
    fontSize: 13,
    border: '1px solid',
    borderColor: active ? '#1677ff' : '#d9d9d9',
    borderRadius: 6,
    background: active ? '#1677ff' : '#fff',
    color: active ? '#fff' : '#333',
    cursor: 'pointer',
    transition: 'all .15s',
  };
}
