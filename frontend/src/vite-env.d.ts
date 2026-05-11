/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE: string;
  readonly VITE_USER_ID: string;
  readonly VITE_SESSION_ID: string;
  readonly VITE_AMAP_KEY: string;
  readonly VITE_AMAP_SECURITY: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
