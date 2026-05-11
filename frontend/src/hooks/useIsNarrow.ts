/**
 * 监听窗口宽度，<breakpoint 视为窄屏（默认 1024px）。
 * 用于切换 Sidebar 是否抽屉化、Chat/Result 是否堆叠。
 */
import { useEffect, useState } from 'react';

export function useIsNarrow(breakpoint = 1024): boolean {
  const [isNarrow, setIsNarrow] = useState(
    typeof window !== 'undefined' ? window.innerWidth < breakpoint : false,
  );

  useEffect(() => {
    const onResize = () => setIsNarrow(window.innerWidth < breakpoint);
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, [breakpoint]);

  return isNarrow;
}
