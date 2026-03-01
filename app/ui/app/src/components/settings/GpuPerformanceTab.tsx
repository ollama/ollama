import { Switch } from "@/components/ui/switch";
import { Select } from "@/components/ui/select";
import { NumberInput } from "@/components/ui/number-input";
import { Field, Label, Description } from "@/components/ui/fieldset";
import {
  BoltIcon,
  CpuChipIcon,
  ServerStackIcon,
  CircleStackIcon,
  ArrowsPointingOutIcon,
  SparklesIcon,
} from "@heroicons/react/20/solid";
import type { Settings } from "@/gotypes";

interface GpuPerformanceTabProps {
  settings: Settings;
  onChange: (field: keyof Settings, value: boolean | string | number | null) => void;
}

const kvCacheOptions = [
  { value: "", label: "Default (f16)" },
  { value: "q8_0", label: "q8_0" },
  { value: "q4_0", label: "q4_0" },
];

export default function GpuPerformanceTab({
  settings,
  onChange,
}: GpuPerformanceTabProps) {
  const isMac = navigator.platform.toLowerCase().includes("mac");

  return (
    <div className="overflow-hidden rounded-xl bg-white dark:bg-neutral-800">
      <div className="space-y-4 p-4">
        {/* Flash Attention */}
        <Field>
          <div className="flex items-start justify-between gap-4">
            <div className="flex items-start space-x-3 flex-1">
              <BoltIcon className="mt-1 h-5 w-5 flex-shrink-0 text-black dark:text-neutral-100" />
              <div>
                <Label>Flash Attention</Label>
                <Description>
                  Enable flash attention for faster inference and lower memory usage.
                </Description>
              </div>
            </div>
            <div className="flex-shrink-0">
              <Switch
                checked={settings.FlashAttention}
                onChange={(checked) => onChange("FlashAttention", checked)}
              />
            </div>
          </div>
        </Field>

        {/* KV Cache Type */}
        <Field>
          <div className="flex items-start space-x-3">
            <CircleStackIcon className="mt-1 h-5 w-5 flex-shrink-0 text-black dark:text-neutral-100" />
            <div className="w-full">
              <Label>KV cache type</Label>
              <Description>
                Quantization type for the KV cache. Lower precision uses less memory but may reduce quality.
              </Description>
              <div className="mt-2">
                <Select
                  value={settings.KvCacheType || ""}
                  onChange={(value) => onChange("KvCacheType", value)}
                  options={kvCacheOptions}
                />
              </div>
            </div>
          </div>
        </Field>

        {/* Parallel Requests */}
        <Field>
          <div className="flex items-start space-x-3">
            <ServerStackIcon className="mt-1 h-5 w-5 flex-shrink-0 text-black dark:text-neutral-100" />
            <div className="w-full">
              <Label>Parallel requests</Label>
              <Description>
                Number of concurrent requests to process. Set to 0 for automatic.
              </Description>
              <div className="mt-2">
                <NumberInput
                  value={settings.NumParallel}
                  onChange={(value) => onChange("NumParallel", value)}
                  min={0}
                  max={16}
                  suffix="requests"
                />
              </div>
            </div>
          </div>
        </Field>

        {/* GPU Overhead */}
        <Field>
          <div className="flex items-start space-x-3">
            <CpuChipIcon className="mt-1 h-5 w-5 flex-shrink-0 text-black dark:text-neutral-100" />
            <div className="w-full">
              <Label>GPU memory overhead</Label>
              <Description>
                Amount of GPU memory to reserve for other applications. Set to 0 for none.
              </Description>
              <div className="mt-2">
                <NumberInput
                  value={settings.GpuOverhead}
                  onChange={(value) => onChange("GpuOverhead", value)}
                  min={0}
                  max={65536}
                  step={256}
                  suffix="MB"
                />
              </div>
            </div>
          </div>
        </Field>

        {/* Schedule Spread */}
        <Field>
          <div className="flex items-start justify-between gap-4">
            <div className="flex items-start space-x-3 flex-1">
              <ArrowsPointingOutIcon className="mt-1 h-5 w-5 flex-shrink-0 text-black dark:text-neutral-100" />
              <div>
                <Label>Schedule spread</Label>
                <Description>
                  Spread model layers across all available GPUs instead of packing onto one.
                </Description>
              </div>
            </div>
            <div className="flex-shrink-0">
              <Switch
                checked={settings.SchedSpread}
                onChange={(checked) => onChange("SchedSpread", checked)}
              />
            </div>
          </div>
        </Field>

        {/* Enable Vulkan - only show on non-macOS */}
        {!isMac && (
          <Field>
            <div className="flex items-start justify-between gap-4">
              <div className="flex items-start space-x-3 flex-1">
                <SparklesIcon className="mt-1 h-5 w-5 flex-shrink-0 text-black dark:text-neutral-100" />
                <div>
                  <Label>Enable Vulkan</Label>
                  <Description>
                    Use Vulkan for GPU acceleration. Requires a Vulkan-compatible GPU.
                  </Description>
                </div>
              </div>
              <div className="flex-shrink-0">
                <Switch
                  checked={settings.EnableVulkan}
                  onChange={(checked) => onChange("EnableVulkan", checked)}
                />
              </div>
            </div>
          </Field>
        )}
      </div>
    </div>
  );
}
