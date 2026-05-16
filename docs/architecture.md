# Expandiffusion Architecture

Expandiffusion is split into a FastAPI backend and a React/Konva frontend.

Backend startup creates the built-in adapter registry, then the plugin manager scans the plugin directory. Each enabled plugin registers adapters or generation postprocessors through a scoped registration context. The registry keeps adapter ownership, prevents duplicate adapter ids, and exposes serializable adapter metadata through `/api/adapters`; plugin status, enablement, and load errors are exposed through `/api/plugins`.

The frontend reads `/api/adapters` as the source of truth for model sources, load controls, generation controls, and generation defaults. Enabled postprocessor plugins can append generation controls to this same schema. The inspector renders those controls through generic schema-driven components, then sends the same model-load and outpaint request payloads used before this refactor.

Design values are centralized in CSS variables in `apps/web/src/index.css`; Konva-specific canvas colors are centralized in `apps/web/src/theme/canvasTheme.ts`.
