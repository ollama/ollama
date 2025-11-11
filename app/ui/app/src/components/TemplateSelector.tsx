import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Field, Label, Description } from "@/components/ui/fieldset";
import { Badge } from "@/components/ui/badge";
import {
  DocumentTextIcon,
  SparklesIcon,
} from "@heroicons/react/20/solid";
import { listTemplates, renderTemplate, type Template } from "@/api";

interface TemplateSelectorProps {
  onTemplateRendered?: (text: string) => void;
}

export default function TemplateSelector({
  onTemplateRendered,
}: TemplateSelectorProps) {
  const [selectedTemplate, setSelectedTemplate] = useState<Template | null>(
    null,
  );
  const [variables, setVariables] = useState<Record<string, string>>({});
  const [showVariablesForm, setShowVariablesForm] = useState(false);

  const { data: templates, isLoading } = useQuery({
    queryKey: ["templates"],
    queryFn: listTemplates,
  });

  const renderMutation = useMutation({
    mutationFn: (params: { templateId: string; variables: Record<string, string> }) =>
      renderTemplate(params.templateId, params.variables),
    onSuccess: (rendered) => {
      if (onTemplateRendered) {
        onTemplateRendered(rendered);
      }
      setShowVariablesForm(false);
      setVariables({});
      setSelectedTemplate(null);
    },
  });

  const handleSelectTemplate = (template: Template) => {
    setSelectedTemplate(template);
    setShowVariablesForm(true);
    // Initialize variables based on template (you might want to parse the template to find variables)
    setVariables({});
  };

  const handleRender = () => {
    if (!selectedTemplate) return;
    renderMutation.mutate({
      templateId: selectedTemplate.id,
      variables,
    });
  };

  // Common template variables that might be needed
  const commonVariables = [
    "code",
    "language",
    "description",
    "file_path",
    "function_name",
    "error_message",
    "context",
  ];

  if (isLoading) {
    return (
      <div className="text-sm text-gray-400">Loading templates...</div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold text-white mb-2">
          Prompt Templates
        </h3>
        <p className="text-sm text-gray-400">
          Use pre-configured templates for common tasks
        </p>
      </div>

      {!showVariablesForm ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {templates && templates.length > 0 ? (
            templates.map((template) => (
              <button
                key={template.id}
                onClick={() => handleSelectTemplate(template)}
                className="text-left p-4 rounded-lg border border-gray-700 bg-gray-800 hover:border-gray-600 hover:bg-gray-750 transition-colors"
              >
                <div className="flex items-start gap-3">
                  <DocumentTextIcon className="h-5 w-5 text-blue-400 flex-shrink-0 mt-1" />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-medium text-white truncate">
                        {template.name}
                      </span>
                      <Badge className="text-xs">{template.id}</Badge>
                    </div>
                    <p className="text-sm text-gray-400 line-clamp-2">
                      {template.description}
                    </p>
                  </div>
                </div>
              </button>
            ))
          ) : (
            <div className="col-span-full text-center py-8 text-gray-400">
              <p>No templates available</p>
            </div>
          )}
        </div>
      ) : (
        <div className="rounded-lg border border-gray-700 bg-gray-800 p-4 space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="font-medium text-white">
              Configure: {selectedTemplate?.name}
            </h4>
            <Button
              onClick={() => {
                setShowVariablesForm(false);
                setSelectedTemplate(null);
                setVariables({});
              }}
              variant="ghost"
              size="sm"
            >
              Cancel
            </Button>
          </div>

          <p className="text-sm text-gray-400">
            {selectedTemplate?.description}
          </p>

          <div className="space-y-3">
            {commonVariables.map((varName) => (
              <Field key={varName}>
                <Label className="capitalize">
                  {varName.replace(/_/g, " ")}
                </Label>
                {varName === "code" || varName === "description" || varName === "context" ? (
                  <textarea
                    className="w-full rounded-md border border-gray-600 bg-gray-700 px-3 py-2 text-white min-h-[100px]"
                    placeholder={`Enter ${varName.replace(/_/g, " ")}...`}
                    value={variables[varName] || ""}
                    onChange={(e) =>
                      setVariables({ ...variables, [varName]: e.target.value })
                    }
                  />
                ) : (
                  <Input
                    type="text"
                    placeholder={`Enter ${varName.replace(/_/g, " ")}...`}
                    value={variables[varName] || ""}
                    onChange={(e) =>
                      setVariables({ ...variables, [varName]: e.target.value })
                    }
                  />
                )}
              </Field>
            ))}
          </div>

          <div className="flex gap-2 justify-end pt-4 border-t border-gray-700">
            <Button
              onClick={handleRender}
              disabled={renderMutation.isPending}
              className="flex items-center gap-2"
            >
              <SparklesIcon className="h-5 w-5" />
              {renderMutation.isPending ? "Rendering..." : "Use Template"}
            </Button>
          </div>
        </div>
      )}

      {renderMutation.isError && (
        <div className="text-sm text-red-500">
          Failed to render template: {renderMutation.error.message}
        </div>
      )}
    </div>
  );
}
