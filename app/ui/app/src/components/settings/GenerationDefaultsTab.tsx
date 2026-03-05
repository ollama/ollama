import type React from "react";
import { Slider } from "@/components/ui/slider";
import { NumberInput } from "@/components/ui/number-input";
import { Button } from "@/components/ui/button";
import { Field, Label, Description } from "@/components/ui/fieldset";
import {
  FireIcon,
  FunnelIcon,
  AdjustmentsHorizontalIcon,
  ArrowPathIcon,
  ArrowUturnLeftIcon,
  KeyIcon,
  DocumentTextIcon,
} from "@heroicons/react/20/solid";
import type { Settings } from "@/gotypes";

interface GenerationDefaultsTabProps {
  settings: Settings;
  onChange: (
    field: keyof Settings,
    value: boolean | string | number | null,
  ) => void;
}

interface SliderParamProps {
  icon: React.ReactNode;
  label: string;
  description: string;
  value: number | null;
  defaultValue: number;
  min: number;
  max: number;
  step: number;
  field: keyof Settings;
  onChange: (
    field: keyof Settings,
    value: boolean | string | number | null,
  ) => void;
}

function SliderParam({
  icon,
  label,
  description,
  value,
  defaultValue,
  min,
  max,
  step,
  field,
  onChange,
}: SliderParamProps) {
  const displayValue = value ?? defaultValue;
  const isDefault = value === null || value === undefined;

  // Build slider options from min to max with the given step
  const optionCount = Math.round((max - min) / step) + 1;
  // For sliders with many steps, show fewer labels
  const maxLabels = 9;
  const labelStep = optionCount > maxLabels ? Math.ceil(optionCount / maxLabels) : 1;

  const options: { value: number; label: string }[] = [];
  for (let i = 0; i < optionCount; i++) {
    const val = Math.round((min + i * step) * 1000) / 1000;
    if (i % labelStep === 0 || i === optionCount - 1) {
      options.push({ value: val, label: String(val) });
    } else {
      options.push({ value: val, label: "" });
    }
  }

  return (
    <Field>
      <div className="flex items-start space-x-3">
        {icon}
        <div className="w-full">
          <div className="flex items-center justify-between">
            <div>
              <Label>{label}</Label>
              <Description>{description}</Description>
            </div>
            <div className="flex items-center space-x-2 flex-shrink-0">
              <span className="text-sm text-zinc-500 dark:text-zinc-400">
                {isDefault ? "Default" : displayValue}
              </span>
              {!isDefault && (
                <Button
                  type="button"
                  color="white"
                  className="px-2 py-1 text-xs"
                  onClick={() => onChange(field, null)}
                >
                  Reset
                </Button>
              )}
            </div>
          </div>
          <div className="mt-3">
            <Slider
              value={displayValue}
              onChange={(val) => onChange(field, val)}
              options={options}
            />
          </div>
        </div>
      </div>
    </Field>
  );
}

interface NumberParamProps {
  icon: React.ReactNode;
  label: string;
  description: string;
  value: number | null;
  defaultValue: number;
  min: number;
  max: number;
  step?: number;
  field: keyof Settings;
  onChange: (
    field: keyof Settings,
    value: boolean | string | number | null,
  ) => void;
}

function NumberParam({
  icon,
  label,
  description,
  value,
  defaultValue,
  min,
  max,
  step = 1,
  field,
  onChange,
}: NumberParamProps) {
  const displayValue = value ?? defaultValue;
  const isDefault = value === null || value === undefined;

  return (
    <Field>
      <div className="flex items-start space-x-3">
        {icon}
        <div className="w-full">
          <div className="flex items-center justify-between">
            <div>
              <Label>{label}</Label>
              <Description>{description}</Description>
            </div>
            <div className="flex items-center space-x-2 flex-shrink-0">
              {isDefault && (
                <span className="text-sm text-zinc-500 dark:text-zinc-400">
                  Default
                </span>
              )}
              {!isDefault && (
                <Button
                  type="button"
                  color="white"
                  className="px-2 py-1 text-xs"
                  onClick={() => onChange(field, null)}
                >
                  Reset
                </Button>
              )}
            </div>
          </div>
          <div className="mt-2">
            <NumberInput
              value={displayValue}
              onChange={(val) => onChange(field, val)}
              min={min}
              max={max}
              step={step}
            />
          </div>
        </div>
      </div>
    </Field>
  );
}

export default function GenerationDefaultsTab({
  settings,
  onChange,
}: GenerationDefaultsTabProps) {
  return (
    <div className="overflow-hidden rounded-xl bg-white dark:bg-neutral-800">
      <div className="space-y-4 p-4">
        {/* Temperature */}
        <SliderParam
          icon={
            <FireIcon className="mt-1 h-5 w-5 flex-shrink-0 text-black dark:text-neutral-100" />
          }
          label="Temperature"
          description="Controls randomness in generation. Higher values produce more creative output."
          value={settings.DefaultTemperature}
          defaultValue={0.8}
          min={0.0}
          max={2.0}
          step={0.05}
          field="DefaultTemperature"
          onChange={onChange}
        />

        {/* Top K */}
        <NumberParam
          icon={
            <FunnelIcon className="mt-1 h-5 w-5 flex-shrink-0 text-black dark:text-neutral-100" />
          }
          label="Top K"
          description="Limits token selection to the K most likely tokens."
          value={settings.DefaultTopK}
          defaultValue={40}
          min={1}
          max={200}
          field="DefaultTopK"
          onChange={onChange}
        />

        {/* Top P */}
        <SliderParam
          icon={
            <AdjustmentsHorizontalIcon className="mt-1 h-5 w-5 flex-shrink-0 text-black dark:text-neutral-100" />
          }
          label="Top P"
          description="Limits token selection to a cumulative probability threshold."
          value={settings.DefaultTopP}
          defaultValue={0.9}
          min={0.0}
          max={1.0}
          step={0.05}
          field="DefaultTopP"
          onChange={onChange}
        />

        {/* Min P */}
        <SliderParam
          icon={
            <AdjustmentsHorizontalIcon className="mt-1 h-5 w-5 flex-shrink-0 text-black dark:text-neutral-100" />
          }
          label="Min P"
          description="Minimum probability threshold relative to the top token."
          value={settings.DefaultMinP}
          defaultValue={0.0}
          min={0.0}
          max={1.0}
          step={0.01}
          field="DefaultMinP"
          onChange={onChange}
        />

        {/* Repeat Penalty */}
        <SliderParam
          icon={
            <ArrowPathIcon className="mt-1 h-5 w-5 flex-shrink-0 text-black dark:text-neutral-100" />
          }
          label="Repeat penalty"
          description="Penalizes repeated tokens to reduce repetitive output."
          value={settings.DefaultRepeatPenalty}
          defaultValue={1.1}
          min={0.0}
          max={2.0}
          step={0.05}
          field="DefaultRepeatPenalty"
          onChange={onChange}
        />

        {/* Repeat Last N */}
        <NumberParam
          icon={
            <ArrowUturnLeftIcon className="mt-1 h-5 w-5 flex-shrink-0 text-black dark:text-neutral-100" />
          }
          label="Repeat last N"
          description="Number of recent tokens to consider for repeat penalty."
          value={settings.DefaultRepeatLastN}
          defaultValue={64}
          min={0}
          max={4096}
          field="DefaultRepeatLastN"
          onChange={onChange}
        />

        {/* Seed */}
        <NumberParam
          icon={
            <KeyIcon className="mt-1 h-5 w-5 flex-shrink-0 text-black dark:text-neutral-100" />
          }
          label="Seed"
          description="Random seed for reproducible generation. Use -1 for random."
          value={settings.DefaultSeed}
          defaultValue={-1}
          min={-1}
          max={999999999}
          field="DefaultSeed"
          onChange={onChange}
        />

        {/* Num Predict */}
        <NumberParam
          icon={
            <DocumentTextIcon className="mt-1 h-5 w-5 flex-shrink-0 text-black dark:text-neutral-100" />
          }
          label="Max tokens"
          description="Maximum number of tokens to generate. Use -1 for unlimited."
          value={settings.DefaultNumPredict}
          defaultValue={-1}
          min={-1}
          max={999999999}
          field="DefaultNumPredict"
          onChange={onChange}
        />
      </div>
    </div>
  );
}
