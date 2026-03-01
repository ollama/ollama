import { Input } from "@/components/ui/input";
import { Field, Label, Description } from "@/components/ui/fieldset";
import {
  ServerIcon,
  GlobeAltIcon,
  ShieldCheckIcon,
  NoSymbolIcon,
  LinkIcon,
  LockClosedIcon,
} from "@heroicons/react/20/solid";
import type { Settings } from "@/gotypes";

interface NetworkSecurityTabProps {
  settings: Settings;
  onChange: (field: keyof Settings, value: boolean | string | number | null) => void;
}

export default function NetworkSecurityTab({
  settings,
  onChange,
}: NetworkSecurityTabProps) {
  return (
    <div className="overflow-hidden rounded-xl bg-white dark:bg-neutral-800">
      <div className="space-y-4 p-4">
        {/* Host/Port */}
        <Field>
          <div className="flex items-start space-x-3">
            <ServerIcon className="mt-1 h-5 w-5 flex-shrink-0 text-black dark:text-neutral-100" />
            <div className="w-full">
              <Label>Host / Port</Label>
              <Description>
                The address and port Ollama listens on.
              </Description>
              <div className="mt-2">
                <Input
                  value={settings.OllamaHost || ""}
                  onChange={(e) => onChange("OllamaHost", e.target.value)}
                  placeholder="127.0.0.1:11434"
                />
              </div>
            </div>
          </div>
        </Field>

        {/* HTTP Proxy */}
        <Field>
          <div className="flex items-start space-x-3">
            <GlobeAltIcon className="mt-1 h-5 w-5 flex-shrink-0 text-black dark:text-neutral-100" />
            <div className="w-full">
              <Label>HTTP proxy</Label>
              <Description>
                Proxy server for HTTP requests.
              </Description>
              <div className="mt-2">
                <Input
                  value={settings.HttpProxy || ""}
                  onChange={(e) => onChange("HttpProxy", e.target.value)}
                  placeholder="http://proxy:8080"
                />
              </div>
            </div>
          </div>
        </Field>

        {/* HTTPS Proxy */}
        <Field>
          <div className="flex items-start space-x-3">
            <LockClosedIcon className="mt-1 h-5 w-5 flex-shrink-0 text-black dark:text-neutral-100" />
            <div className="w-full">
              <Label>HTTPS proxy</Label>
              <Description>
                Proxy server for HTTPS requests.
              </Description>
              <div className="mt-2">
                <Input
                  value={settings.HttpsProxy || ""}
                  onChange={(e) => onChange("HttpsProxy", e.target.value)}
                  placeholder="https://proxy:8443"
                />
              </div>
            </div>
          </div>
        </Field>

        {/* No Proxy */}
        <Field>
          <div className="flex items-start space-x-3">
            <NoSymbolIcon className="mt-1 h-5 w-5 flex-shrink-0 text-black dark:text-neutral-100" />
            <div className="w-full">
              <Label>No proxy</Label>
              <Description>
                Comma-separated list of hosts that should bypass the proxy.
              </Description>
              <div className="mt-2">
                <Input
                  value={settings.NoProxy || ""}
                  onChange={(e) => onChange("NoProxy", e.target.value)}
                  placeholder="localhost,127.0.0.1"
                />
              </div>
            </div>
          </div>
        </Field>

        {/* CORS Origins */}
        <Field>
          <div className="flex items-start space-x-3">
            <LinkIcon className="mt-1 h-5 w-5 flex-shrink-0 text-black dark:text-neutral-100" />
            <div className="w-full">
              <Label>CORS origins</Label>
              <Description>
                Allowed origins for cross-origin requests.
              </Description>
              <div className="mt-2">
                <Input
                  value={settings.CorsOrigins || ""}
                  onChange={(e) => onChange("CorsOrigins", e.target.value)}
                  placeholder="http://localhost:3000"
                />
              </div>
            </div>
          </div>
        </Field>

        {/* Allowed Remotes */}
        <Field>
          <div className="flex items-start space-x-3">
            <ShieldCheckIcon className="mt-1 h-5 w-5 flex-shrink-0 text-black dark:text-neutral-100" />
            <div className="w-full">
              <Label>Allowed remotes</Label>
              <Description>
                Restrict which remote hosts can connect to Ollama.
              </Description>
              <div className="mt-2">
                <Input
                  value={settings.AllowedRemotes || ""}
                  onChange={(e) => onChange("AllowedRemotes", e.target.value)}
                  placeholder="ollama.com"
                />
              </div>
            </div>
          </div>
        </Field>
      </div>
    </div>
  );
}
