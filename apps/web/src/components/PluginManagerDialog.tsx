import { useMutation } from '@tanstack/react-query'
import { X } from 'lucide-react'
import { useMemo } from 'react'
import { disablePlugin, enablePlugin } from '../lib/apiClient'
import { useStudioQueries } from '../hooks/useStudioQueries'
import { useEditorStore } from '../store/editorStore'
import { PluginManagerSection } from './PluginManagerSection'
import { Button } from './ui/button'

interface PluginManagerDialogProps {
  onClose: () => void
}

/**
 * Render plugin management outside the right inspector.
 *
 * @param props - Dialog close callback.
 * @returns Plugin manager dialog.
 */
export function PluginManagerDialog({ onClose }: PluginManagerDialogProps) {
  const {
    adaptersQuery,
    pluginsQuery,
    pluginActionsQuery,
    pluginToolsQuery,
    modelsQuery,
    persistentStateQuery,
  } = useStudioQueries()
  const selectedAdapterId = useEditorStore((state) => state.selectedAdapterId)
  const parameters = useEditorStore((state) => state.parameters)
  const updateParameter = useEditorStore((state) => state.updateParameter)
  const setErrorMessage = useEditorStore((state) => state.setErrorMessage)

  const adapters = useMemo(() => adaptersQuery.data ?? [], [adaptersQuery.data])
  const selectedAdapter = useMemo(
    () => adapters.find((adapter) => adapter.id === selectedAdapterId) ?? adapters[0],
    [adapters, selectedAdapterId],
  )

  const pluginMutation = useMutation({
    mutationFn: ({ pluginId, enabled }: { pluginId: string; enabled: boolean }) =>
      enabled ? enablePlugin(pluginId) : disablePlugin(pluginId),
    onSuccess: async () => {
      await pluginsQuery.refetch()
      await adaptersQuery.refetch()
      await pluginActionsQuery.refetch()
      await pluginToolsQuery.refetch()
      await modelsQuery.refetch()
      await persistentStateQuery.refetch()
    },
    onError: (error) => {
      setErrorMessage(error instanceof Error ? error.message : 'Plugin update failed.')
    },
  })

  const enabledCount = pluginsQuery.data?.filter((plugin) => plugin.enabled).length ?? 0
  const pluginCount = pluginsQuery.data?.length ?? 0

  return (
    <div
      className="setup-overlay"
      role="dialog"
      aria-modal="true"
      aria-label="Plugins"
    >
      <div className="setup-dialog plugin-manager-dialog">
        <div className="setup-header">
          <div>
            <h2>Plugins</h2>
            <p>{enabledCount} / {pluginCount} enabled</p>
          </div>
          <Button
            type="button"
            variant="ghost"
            size="smallIcon"
            className="setup-close-button"
            aria-label="Close dialog"
            title="Close"
            onClick={onClose}
          >
            <X size={16} />
          </Button>
        </div>
        <PluginManagerSection
          plugins={pluginsQuery.data ?? []}
          controls={selectedAdapter?.generation_controls ?? []}
          postprocessors={selectedAdapter?.postprocessors ?? []}
          parameters={parameters}
          pendingPluginId={
            pluginMutation.isPending && pluginMutation.variables
              ? pluginMutation.variables.pluginId
              : null
          }
          onToggle={(plugin) =>
            pluginMutation.mutate({
              pluginId: plugin.id,
              enabled: !plugin.enabled,
            })
          }
          onParameterChange={updateParameter}
        />
      </div>
    </div>
  )
}
