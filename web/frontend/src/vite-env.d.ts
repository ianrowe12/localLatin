/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly BASE_URL: string
  readonly VITE_API_BASE_PATH?: string
  readonly VITE_BASE_PATH?: string
  readonly VITE_USE_MOCKS: string
}
interface ImportMeta {
  readonly env: ImportMetaEnv
}
