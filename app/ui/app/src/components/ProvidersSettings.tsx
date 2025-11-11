import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Field, Label, Description } from "@/components/ui/fieldset";
import { Badge } from "@/components/ui/badge";
import {
  PlusIcon,
  TrashIcon,
  CheckCircleIcon,
  XCircleIcon,
} from "@heroicons/react/20/solid";
import {
  listProviders,
  addProvider,
  deleteProvider,
  validateProvider,
  type Provider,
} from "@/api";

export default function ProvidersSettings() {
  const queryClient = useQueryClient();
  const [showAddForm, setShowAddForm] = useState(false);
  const [newProvider, setNewProvider] = useState<Provider>({
    type: "openai",
    name: "",
    api_key: "",
    base_url: "",
  });
  const [validationStatus, setValidationStatus] = useState<{
    validating: boolean;
    valid?: boolean;
    error?: string;
  }>({ validating: false });

  const { data: providers, isLoading } = useQuery({
    queryKey: ["providers"],
    queryFn: listProviders,
  });

  const addProviderMutation = useMutation({
    mutationFn: addProvider,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["providers"] });
      setShowAddForm(false);
      setNewProvider({
        type: "openai",
        name: "",
        api_key: "",
        base_url: "",
      });
      setValidationStatus({ validating: false });
    },
  });

  const deleteProviderMutation = useMutation({
    mutationFn: deleteProvider,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["providers"] });
    },
  });

  const handleValidate = async () => {
    if (!newProvider.api_key) return;

    setValidationStatus({ validating: true });
    try {
      const valid = await validateProvider(
        newProvider.type,
        newProvider.api_key,
      );
      setValidationStatus({ validating: false, valid });
    } catch (error) {
      setValidationStatus({
        validating: false,
        valid: false,
        error: error instanceof Error ? error.message : "Validation failed",
      });
    }
  };

  const handleAddProvider = () => {
    if (!newProvider.name || !newProvider.api_key) {
      alert("Please fill in all required fields");
      return;
    }

    addProviderMutation.mutate(newProvider);
  };

  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="text-sm text-gray-500">Loading providers...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-white">API Providers</h3>
          <p className="text-sm text-gray-400">
            Manage external AI service providers
          </p>
        </div>
        <Button
          onClick={() => setShowAddForm(!showAddForm)}
          className="flex items-center gap-2"
        >
          <PlusIcon className="h-5 w-5" />
          Add Provider
        </Button>
      </div>

      {showAddForm && (
        <div className="rounded-lg border border-gray-700 bg-gray-800 p-4 space-y-4">
          <h4 className="font-medium text-white">Add New Provider</h4>

          <Field>
            <Label>Provider Type</Label>
            <select
              value={newProvider.type}
              onChange={(e) =>
                setNewProvider({ ...newProvider, type: e.target.value })
              }
              className="w-full rounded-md border border-gray-600 bg-gray-700 px-3 py-2 text-white"
            >
              <option value="openai">OpenAI</option>
              <option value="anthropic">Anthropic</option>
              <option value="google">Google Gemini</option>
              <option value="groq">Groq</option>
            </select>
          </Field>

          <Field>
            <Label>Provider Name</Label>
            <Input
              type="text"
              placeholder="My OpenAI Account"
              value={newProvider.name}
              onChange={(e) =>
                setNewProvider({ ...newProvider, name: e.target.value })
              }
            />
            <Description>A friendly name to identify this provider</Description>
          </Field>

          <Field>
            <Label>API Key</Label>
            <div className="flex gap-2">
              <Input
                type="password"
                placeholder="sk-..."
                value={newProvider.api_key}
                onChange={(e) =>
                  setNewProvider({ ...newProvider, api_key: e.target.value })
                }
                className="flex-1"
              />
              <Button onClick={handleValidate} disabled={validationStatus.validating}>
                {validationStatus.validating ? "Validating..." : "Validate"}
              </Button>
            </div>
            {validationStatus.valid !== undefined && (
              <div className="flex items-center gap-2 mt-2">
                {validationStatus.valid ? (
                  <>
                    <CheckCircleIcon className="h-5 w-5 text-green-500" />
                    <span className="text-sm text-green-500">Valid API key</span>
                  </>
                ) : (
                  <>
                    <XCircleIcon className="h-5 w-5 text-red-500" />
                    <span className="text-sm text-red-500">
                      Invalid API key {validationStatus.error && `- ${validationStatus.error}`}
                    </span>
                  </>
                )}
              </div>
            )}
          </Field>

          {newProvider.type === "openai" && (
            <Field>
              <Label>Base URL (Optional)</Label>
              <Input
                type="text"
                placeholder="https://api.openai.com/v1"
                value={newProvider.base_url}
                onChange={(e) =>
                  setNewProvider({ ...newProvider, base_url: e.target.value })
                }
              />
              <Description>
                Use a custom API endpoint (for OpenAI-compatible APIs)
              </Description>
            </Field>
          )}

          <div className="flex gap-2 justify-end">
            <Button
              onClick={() => {
                setShowAddForm(false);
                setValidationStatus({ validating: false });
              }}
              variant="outline"
            >
              Cancel
            </Button>
            <Button
              onClick={handleAddProvider}
              disabled={addProviderMutation.isPending}
            >
              {addProviderMutation.isPending ? "Adding..." : "Add Provider"}
            </Button>
          </div>
        </div>
      )}

      <div className="space-y-3">
        {providers && providers.length > 0 ? (
          providers.map((provider) => (
            <div
              key={provider.type}
              className="flex items-center justify-between rounded-lg border border-gray-700 bg-gray-800 p-4"
            >
              <div className="flex items-center gap-3">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-white">
                      {provider.name}
                    </span>
                    <Badge>{provider.type}</Badge>
                  </div>
                  {provider.base_url && (
                    <p className="text-xs text-gray-400 mt-1">
                      Custom URL: {provider.base_url}
                    </p>
                  )}
                </div>
              </div>
              <Button
                onClick={() => deleteProviderMutation.mutate(provider.type)}
                variant="ghost"
                className="text-red-500 hover:text-red-400"
              >
                <TrashIcon className="h-5 w-5" />
              </Button>
            </div>
          ))
        ) : (
          <div className="text-center py-8 text-gray-400">
            <p>No providers configured</p>
            <p className="text-sm mt-2">
              Add a provider to start using external AI services
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
