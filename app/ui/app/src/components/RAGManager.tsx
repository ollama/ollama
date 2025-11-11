import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Field, Label, Description } from "@/components/ui/fieldset";
import {
  DocumentPlusIcon,
  MagnifyingGlassIcon,
  CheckCircleIcon,
} from "@heroicons/react/20/solid";
import { ingestDocument, searchRAG, type RAGSearchResult } from "@/api";

interface RAGManagerProps {
  workspacePath: string;
}

export default function RAGManager({ workspacePath }: RAGManagerProps) {
  const [showIngestForm, setShowIngestForm] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<RAGSearchResult[]>([]);
  const [newDoc, setNewDoc] = useState({
    title: "",
    content: "",
  });

  const ingestMutation = useMutation({
    mutationFn: (doc: { title: string; content: string }) =>
      ingestDocument(workspacePath, doc.title, doc.content),
    onSuccess: () => {
      setShowIngestForm(false);
      setNewDoc({ title: "", content: "" });
      alert("Document ingested successfully!");
    },
  });

  const searchMutation = useMutation({
    mutationFn: (query: string) => searchRAG(workspacePath, query, 5),
    onSuccess: (results) => {
      setSearchResults(results);
    },
  });

  const handleIngest = () => {
    if (!newDoc.title || !newDoc.content) {
      alert("Please fill in all fields");
      return;
    }
    ingestMutation.mutate(newDoc);
  };

  const handleSearch = () => {
    if (!searchQuery) return;
    searchMutation.mutate(searchQuery);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-white">RAG Document Management</h3>
          <p className="text-sm text-gray-400">
            Ingest and search documents for context-aware responses
          </p>
        </div>
        <Button
          onClick={() => setShowIngestForm(!showIngestForm)}
          className="flex items-center gap-2"
        >
          <DocumentPlusIcon className="h-5 w-5" />
          Ingest Document
        </Button>
      </div>

      {showIngestForm && (
        <div className="rounded-lg border border-gray-700 bg-gray-800 p-4 space-y-4">
          <h4 className="font-medium text-white">Ingest New Document</h4>

          <Field>
            <Label>Document Title</Label>
            <Input
              type="text"
              placeholder="API Documentation"
              value={newDoc.title}
              onChange={(e) => setNewDoc({ ...newDoc, title: e.target.value })}
            />
          </Field>

          <Field>
            <Label>Content</Label>
            <textarea
              className="w-full rounded-md border border-gray-600 bg-gray-700 px-3 py-2 text-white min-h-[200px]"
              placeholder="Paste your document content here..."
              value={newDoc.content}
              onChange={(e) => setNewDoc({ ...newDoc, content: e.target.value })}
            />
            <Description>
              Large documents will be automatically chunked for efficient retrieval
            </Description>
          </Field>

          <div className="flex gap-2 justify-end">
            <Button
              onClick={() => setShowIngestForm(false)}
              variant="outline"
            >
              Cancel
            </Button>
            <Button
              onClick={handleIngest}
              disabled={ingestMutation.isPending}
            >
              {ingestMutation.isPending ? "Ingesting..." : "Ingest"}
            </Button>
          </div>

          {ingestMutation.isSuccess && (
            <div className="flex items-center gap-2 text-green-500">
              <CheckCircleIcon className="h-5 w-5" />
              <span className="text-sm">Document ingested successfully!</span>
            </div>
          )}
        </div>
      )}

      {/* Search Section */}
      <div className="space-y-4">
        <h4 className="font-medium text-white">Search Documents</h4>
        <div className="flex gap-2">
          <Input
            type="text"
            placeholder="Search for relevant information..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSearch()}
            className="flex-1"
          />
          <Button
            onClick={handleSearch}
            disabled={searchMutation.isPending}
            className="flex items-center gap-2"
          >
            <MagnifyingGlassIcon className="h-5 w-5" />
            Search
          </Button>
        </div>

        {searchMutation.isPending && (
          <div className="text-sm text-gray-400">Searching...</div>
        )}

        {searchResults.length > 0 && (
          <div className="space-y-3">
            <p className="text-sm text-gray-400">
              Found {searchResults.length} relevant chunks:
            </p>
            {searchResults.map((result, idx) => (
              <div
                key={idx}
                className="rounded-lg border border-gray-700 bg-gray-800 p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs text-gray-400">
                    Relevance: {(result.score * 100).toFixed(1)}%
                  </span>
                </div>
                <p className="text-sm text-white whitespace-pre-wrap">
                  {result.chunk}
                </p>
                {result.metadata && Object.keys(result.metadata).length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-2">
                    {Object.entries(result.metadata).map(([key, value]) => (
                      <span
                        key={key}
                        className="text-xs bg-gray-700 px-2 py-1 rounded"
                      >
                        {key}: {value}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
