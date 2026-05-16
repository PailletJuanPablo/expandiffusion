import {
  ADAPTER_SD15_CONTROLNET_INPAINT,
  ADAPTER_SD15_INPAINT,
  ADAPTER_SDXL_CONTROLNET_INPAINT,
  ADAPTER_SDXL_INPAINT,
} from "../constants/domain";
import type { AdapterInfo } from "../domain/types";

const CONTROLNET_ADAPTER_BY_STANDARD_ADAPTER: Record<string, string> = {
  [ADAPTER_SD15_INPAINT]: ADAPTER_SD15_CONTROLNET_INPAINT,
  [ADAPTER_SDXL_INPAINT]: ADAPTER_SDXL_CONTROLNET_INPAINT,
};

export function adapterIdForControlGuideMode(
  enabled: boolean,
  selectedAdapterId: string,
  adapters: AdapterInfo[],
): string | null {
  if (!enabled) {
    return null;
  }
  const selectedAdapter = adapters.find((adapter) => adapter.id === selectedAdapterId);
  if (!selectedAdapter || selectedAdapter.capabilities.controlnet) {
    return null;
  }
  const pairedAdapterId = CONTROLNET_ADAPTER_BY_STANDARD_ADAPTER[selectedAdapter.id];
  const pairedAdapter = pairedAdapterId
    ? adapters.find((adapter) => adapter.id === pairedAdapterId)
    : null;
  if (pairedAdapter?.capabilities.controlnet) {
    return pairedAdapter.id;
  }
  const familyMatch = adapters.find(
    (adapter) =>
      adapter.capabilities.controlnet &&
      adapter.family === selectedAdapter.family &&
      adapter.capabilities.inpaint === selectedAdapter.capabilities.inpaint &&
      adapter.capabilities.outpaint === selectedAdapter.capabilities.outpaint,
  );
  return familyMatch?.id ?? null;
}

export function controlnetModelIdForAdapter(adapter: AdapterInfo | undefined): string | null {
  const control = adapter?.load_controls.find(
    (item) => item.id === "controlnet_model_id",
  );
  return typeof control?.default_value === "string" ? control.default_value : null;
}
