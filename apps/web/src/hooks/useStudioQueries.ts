import { useQuery } from '@tanstack/react-query'
import {
  getPersistentState,
  getRuntime,
  listAdapters,
  listModels,
  listPluginActions,
  listPlugins,
  listPluginTools,
} from '../lib/apiClient'
import { useI18n } from '../i18n/useI18n'
import {
  localizeAdapterInfo,
  localizeModelInfo,
  localizePersistentState,
  localizePluginActionInfo,
  localizePluginInfo,
  localizePluginToolInfo,
} from '../i18n/metadata'

/**
 * Load studio metadata shared by inspector panels.
 *
 * @returns Query objects for runtime, adapters, models, plugins and persisted state.
 */
export function useStudioQueries() {
  const { t } = useI18n()
  const runtimeQuery = useQuery({ queryKey: ['runtime'], queryFn: getRuntime })
  const adaptersQuery = useQuery({
    queryKey: ['adapters'],
    queryFn: listAdapters,
    select: (adapters) => adapters.map((adapter) => localizeAdapterInfo(adapter, t)),
  })
  const pluginsQuery = useQuery({
    queryKey: ['plugins'],
    queryFn: listPlugins,
    select: (plugins) => plugins.map((plugin) => localizePluginInfo(plugin, t)),
  })
  const pluginActionsQuery = useQuery({
    queryKey: ['plugin-actions'],
    queryFn: listPluginActions,
    select: (actions) => actions.map((action) => localizePluginActionInfo(action, t)),
  })
  const pluginToolsQuery = useQuery({
    queryKey: ['plugin-tools'],
    queryFn: listPluginTools,
    select: (tools) => tools.map((tool) => localizePluginToolInfo(tool, t)),
  })
  const modelsQuery = useQuery({
    queryKey: ['models'],
    queryFn: listModels,
    select: (models) => models.map((model) => localizeModelInfo(model, t)),
  })
  const persistentStateQuery = useQuery({
    queryKey: ['persistent-state'],
    queryFn: getPersistentState,
    select: (state) => localizePersistentState(state, t),
    enabled: false,
    retry: false,
  })

  return {
    runtimeQuery,
    adaptersQuery,
    pluginsQuery,
    pluginActionsQuery,
    pluginToolsQuery,
    modelsQuery,
    persistentStateQuery,
  }
}
