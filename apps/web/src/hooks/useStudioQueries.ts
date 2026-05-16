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

/**
 * Load studio metadata shared by inspector panels.
 *
 * @returns Query objects for runtime, adapters, models, plugins and persisted state.
 */
export function useStudioQueries() {
  const runtimeQuery = useQuery({ queryKey: ['runtime'], queryFn: getRuntime })
  const adaptersQuery = useQuery({ queryKey: ['adapters'], queryFn: listAdapters })
  const pluginsQuery = useQuery({ queryKey: ['plugins'], queryFn: listPlugins })
  const pluginActionsQuery = useQuery({ queryKey: ['plugin-actions'], queryFn: listPluginActions })
  const pluginToolsQuery = useQuery({ queryKey: ['plugin-tools'], queryFn: listPluginTools })
  const modelsQuery = useQuery({ queryKey: ['models'], queryFn: listModels })
  const persistentStateQuery = useQuery({
    queryKey: ['persistent-state'],
    queryFn: getPersistentState,
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
