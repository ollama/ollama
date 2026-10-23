import { BoltIcon } from "@heroicons/react/20/solid";
import { Field, Label, Description } from "@/components/ui/fieldset";
import { Switch } from "@/components/ui/switch";
import {
  defaultDraftNumPredict,
  draftDepthChoices,
  draftNumPredictChoices,
} from "@/lib/mtp";

interface ModelMTPSettingProps {
  modelName: string;
  supported: boolean;
  draftNumPredict?: number;
  pending?: boolean;
  error?: string | null;
  onChange: (draftNumPredict: number) => void;
}

export function ModelMTPSetting({
  modelName,
  supported,
  draftNumPredict,
  pending = false,
  error,
  onChange,
}: ModelMTPSettingProps) {
  if (!supported || !modelName) return null;

  const enabled = (draftNumPredict ?? 0) > 0;
  const depth =
    draftNumPredict && draftNumPredict > 0
      ? draftNumPredict
      : defaultDraftNumPredict;

  return (
    <Field>
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-1 items-start space-x-3">
          <BoltIcon className="mt-1 h-5 w-5 flex-shrink-0 text-black dark:text-neutral-100" />
          <div className="w-full">
            <Label>Multi-token prediction</Label>
            <Description>
              Draft the next few tokens while {modelName} generates. This can be
              faster and uses more memory. The model reloads if it is loaded.
            </Description>
            {enabled && (
              <label className="mt-3 flex items-center gap-2 text-sm text-neutral-600 dark:text-neutral-300">
                <span>Draft tokens</span>
                <select
                  data-testid="mtp-draft-depth"
                  aria-label="Draft tokens"
                  className="rounded-lg border border-neutral-200 bg-white px-2 py-1 text-sm text-neutral-900 dark:border-neutral-700 dark:bg-neutral-900 dark:text-neutral-100"
                  value={depth}
                  disabled={pending}
                  onChange={(event) => {
                    const next = Number(event.target.value);
                    if (
                      !draftNumPredictChoices.includes(next) ||
                      next === draftNumPredict
                    ) {
                      return;
                    }
                    onChange(next);
                  }}
                >
                  {draftDepthChoices(draftNumPredict).map((choice) => (
                    <option key={choice} value={choice}>
                      {choice}
                    </option>
                  ))}
                </select>
              </label>
            )}
            {error && (
              <p
                role="alert"
                data-testid="mtp-error"
                className="mt-2 text-xs text-red-600 dark:text-red-400"
              >
                {error}
              </p>
            )}
          </div>
        </div>
        <div className="flex-shrink-0">
          <Switch
            data-testid="mtp-toggle"
            aria-label={`Multi-token prediction for ${modelName}`}
            checked={enabled}
            disabled={pending}
            onChange={(checked) => {
              if (checked) {
                onChange(
                  enabled && draftNumPredict
                    ? draftNumPredict
                    : defaultDraftNumPredict,
                );
                return;
              }
              onChange(0);
            }}
          />
        </div>
      </div>
    </Field>
  );
}
