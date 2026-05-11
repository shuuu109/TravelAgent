import { useEffect, useState } from 'react';
import AMapLoader from '@amap/amap-jsapi-loader';

// 全局只加载一次：多次 mount AmapView 时共享同一份 AMap
let cachedAMap: any = null;
let loadingPromise: Promise<any> | null = null;

interface UseAmapResult {
  AMap: any | null;
  loading: boolean;
  error: string | null;
}

// 加载高德 JSAPI。Key/SecurityCode 从 .env 注入：
//   VITE_AMAP_KEY            - 必填
//   VITE_AMAP_SECURITY       - 必填（安全密钥；缺失时只能加载部分服务）
export function useAmap(plugins: string[] = []): UseAmapResult {
  const [AMap, setAMap] = useState<any | null>(cachedAMap);
  const [loading, setLoading] = useState<boolean>(!cachedAMap);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (cachedAMap) {
      setAMap(cachedAMap);
      setLoading(false);
      return;
    }

    const key = import.meta.env.VITE_AMAP_KEY;
    const security = import.meta.env.VITE_AMAP_SECURITY;
    if (!key) {
      setError('未配置 VITE_AMAP_KEY，无法加载地图');
      setLoading(false);
      return;
    }

    // 安全密钥：JSAPI loader 要求挂在 window._AMapSecurityConfig
    if (security) {
      (window as any)._AMapSecurityConfig = { securityJsCode: security };
    }

    if (!loadingPromise) {
      loadingPromise = AMapLoader.load({
        key,
        version: '2.0',
        plugins,
      });
    }

    let cancelled = false;
    loadingPromise
      .then((amap) => {
        if (cancelled) return;
        cachedAMap = amap;
        setAMap(amap);
        setLoading(false);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(`地图加载失败：${e?.message || e}`);
        setLoading(false);
      });

    return () => {
      cancelled = true;
    };
    // 仅首次加载使用，plugins 变更不重新加载（@amap/amap-jsapi-loader 不支持二次加载）
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { AMap, loading, error };
}
