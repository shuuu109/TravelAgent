import { useEffect, useRef } from 'react';
import { useAmap } from '../../hooks/useAmap';
import type { DailyRoute } from '../../types/sse';

interface Props {
  route: DailyRoute;
  poiDescriptions?: Record<string, string>;
  poiPhotos?: Record<string, string[]>;
  height?: number;
}

// 单天的地图视图：按 ordered_pois 顺序打编号 marker，并用 Polyline 顺序连线。
// 第一期：不调 Driving / Walking / Transfer 重新规划真实路径，纯几何连线（性能好、绝不出错）。
//
// 切换 day 时仅 mutate 现有 map 上的 overlays，不重建 map 实例 —— 避免重复加载脚本和闪烁。
export default function AmapView({
  route,
  poiDescriptions,
  poiPhotos,
  height = 360,
}: Props) {
  const { AMap, loading, error } = useAmap();
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<any>(null);
  // 缓存当前 day 加在 map 上的 overlays，便于切换时整体清除
  const overlaysRef = useRef<any[]>([]);
  const infoWindowRef = useRef<any>(null);

  // 1) 第一次拿到 AMap + container 就建 map（仅一次）
  useEffect(() => {
    if (!AMap || !containerRef.current || mapRef.current) return;
    mapRef.current = new AMap.Map(containerRef.current, {
      zoom: 12,
      viewMode: '2D',
    });
    infoWindowRef.current = new AMap.InfoWindow({
      offset: new AMap.Pixel(0, -32),
      anchor: 'bottom-center',
    });

    return () => {
      // 严格模式下父组件会卸载-重挂，destroy 会让 map 容器 DOM 变白；
      // 这里只清 overlays，map 实例随页面生命周期销毁
      overlaysRef.current.forEach((o) => o.setMap && o.setMap(null));
      overlaysRef.current = [];
    };
  }, [AMap]);

  // 2) route 变化时重绘 markers + polyline + setFitView
  useEffect(() => {
    if (!AMap || !mapRef.current) return;
    const map = mapRef.current;

    // 清除旧 overlays
    overlaysRef.current.forEach((o) => o.setMap && o.setMap(null));
    overlaysRef.current = [];
    infoWindowRef.current?.close();

    const pois = (route.ordered_pois || []).filter(
      (p) => typeof p.lng === 'number' && typeof p.lat === 'number',
    );
    if (pois.length === 0) return;

    // 编号 markers：用 content 完全替换默认图标（避免出现白底 label 框）
    pois.forEach((poi, idx) => {
      const dom = document.createElement('div');
      dom.style.cssText = [
        'width:26px',
        'height:26px',
        'border-radius:50%',
        'background:#1677ff',
        'color:#fff',
        'display:flex',
        'align-items:center',
        'justify-content:center',
        'font-size:13px',
        'font-weight:600',
        'box-shadow:0 2px 6px rgba(22,119,255,0.45)',
        'border:2px solid #fff',
        'cursor:pointer',
      ].join(';');
      dom.textContent = String(idx + 1);

      const marker = new AMap.Marker({
        position: [poi.lng, poi.lat],
        title: poi.name,
        content: dom,
        anchor: 'center',
        offset: new AMap.Pixel(0, 0),
      });
      marker.setMap(map);
      marker.on('click', () => openInfo(poi));
      overlaysRef.current.push(marker);
    });

    // 连线
    if (pois.length >= 2) {
      const polyline = new AMap.Polyline({
        path: pois.map((p) => [p.lng, p.lat]),
        strokeColor: '#1677ff',
        strokeWeight: 4,
        strokeOpacity: 0.7,
        showDir: true,
      });
      polyline.setMap(map);
      overlaysRef.current.push(polyline);
    }

    // 自动缩放到当天所有点
    map.setFitView(overlaysRef.current, false, [40, 40, 40, 40]);

    function openInfo(poi: any) {
      const desc = (poiDescriptions || {})[poi.name] || '';
      const photo = (poiPhotos || {})[poi.name]?.[0] || '';
      const html = `
        <div style="max-width:240px;font-size:12px;line-height:1.5;">
          <div style="font-weight:600;font-size:13px;margin-bottom:4px;">${escapeHtml(poi.name)}</div>
          ${
            photo
              ? `<img src="${escapeHtml(photo)}" style="width:100%;max-height:120px;object-fit:cover;border-radius:4px;margin-bottom:6px;" />`
              : ''
          }
          ${desc ? `<div style="color:#555;">${escapeHtml(desc)}</div>` : ''}
          ${
            poi.address
              ? `<div style="color:#999;margin-top:4px;font-size:11px;">${escapeHtml(poi.address)}</div>`
              : ''
          }
        </div>
      `;
      infoWindowRef.current.setContent(html);
      infoWindowRef.current.open(map, [poi.lng, poi.lat]);
    }
    // poiDescriptions/poiPhotos 仅在 marker click 时读取最新值；这里依赖 route 即可
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [AMap, route]);

  if (error) {
    return (
      <div style={{ ...wrapperStyle(height), color: '#c00', padding: 12 }}>{error}</div>
    );
  }

  return (
    <div style={{ marginTop: 16 }}>
      <div
        style={{
          fontWeight: 700,
          marginBottom: 8,
          fontSize: 15,
          color: '#1f1f1f',
          borderLeft: '3px solid #52c41a',
          paddingLeft: 8,
          lineHeight: '18px',
        }}
      >
        地图
      </div>
      <div ref={containerRef} style={wrapperStyle(height)}>
        {loading && (
          <div style={{ padding: 16, color: '#999' }}>地图加载中...</div>
        )}
      </div>
    </div>
  );
}

function wrapperStyle(height: number): React.CSSProperties {
  return {
    width: '100%',
    height,
    border: '1px solid #e6e8eb',
    borderRadius: 8,
    overflow: 'hidden',
    background: '#f5f7fa',
  };
}

function escapeHtml(s: string): string {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}
