import {
  useState,
  useCallback,
  useRef,
  useEffect,
  type ReactNode,
} from "react";
import { DocumentPlusIcon } from "@heroicons/react/24/outline";
import type { Model } from "@/gotypes";

const TEXT_FILE_EXTENSIONS = [
  "pdf",
  "docx",
  "txt",
  "md",
  "csv",
  "json",
  "xml",
  "html",
  "htm",
  "js",
  "jsx",
  "ts",
  "tsx",
  "py",
  "java",
  "cpp",
  "c",
  "cc",
  "h",
  "cs",
  "php",
  "rb",
  "go",
  "rs",
  "swift",
  "kt",
  "scala",
  "sh",
  "bat",
  "yaml",
  "yml",
  "toml",
  "ini",
  "cfg",
  "conf",
  "log",
  "rtf",
];

const IMAGE_EXTENSIONS = ["png", "jpg", "jpeg"];

interface FileUploadProps {
  children: ReactNode;
  onFilesAdded: (
    files: Array<{ filename: string; data: Uint8Array; type?: string }>,
    errors: Array<{ filename: string; error: string }>,
  ) => void;
  selectedModel?: Model | null;
  hasVisionCapability?: boolean;
  validateFile?: (file: File) => { valid: boolean; error?: string };
  maxFileSize?: number;
  allowedExtensions?: string[];
}

export function FileUpload({
  children,
  onFilesAdded,
  selectedModel,
  hasVisionCapability = false,
  validateFile,
  maxFileSize = 10,
  allowedExtensions,
}: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  // Counter to track drag enter/leave events across all child elements
  // Prevents flickering when dragging over child elements within the component
  const dragCounter = useRef(0);

  const MAX_FILE_SIZE = maxFileSize * 1024 * 1024; // Convert MB to bytes
  const ALLOWED_EXTENSIONS = allowedExtensions || [
    ...TEXT_FILE_EXTENSIONS,
    ...IMAGE_EXTENSIONS,
  ];

  // Helper function to check if dragging files
  const hasFiles = useCallback((dataTransfer: DataTransfer) => {
    return dataTransfer.types.includes("Files");
  }, []);

  // Helper function to read file as Uint8Array
  const readFileAsBytes = useCallback((file: File): Promise<Uint8Array> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const arrayBuffer = reader.result as ArrayBuffer;
        resolve(new Uint8Array(arrayBuffer));
      };
      reader.onerror = () => reject(reader.error);
      reader.readAsArrayBuffer(file);
    });
  }, []);

  // Helper function to read directory contents
  const readDirectory = useCallback(
    async (entry: FileSystemDirectoryEntry): Promise<File[]> => {
      const files: File[] = [];

      const readEntries = async (
        dirEntry: FileSystemDirectoryEntry,
      ): Promise<void> => {
        const dirReader = dirEntry.createReader();

        return new Promise((resolve, reject) => {
          dirReader.readEntries(async (entries) => {
            try {
              for (const entry of entries) {
                if (entry.isFile) {
                  const fileEntry = entry as FileSystemFileEntry;
                  const file = await new Promise<File>(
                    (resolveFile, rejectFile) => {
                      fileEntry.file(resolveFile, rejectFile);
                    },
                  );
                  files.push(file);
                } else if (entry.isDirectory) {
                  // Skip subdirectories for simplicity
                }
              }
              resolve();
            } catch (error) {
              reject(error);
            }
          }, reject);
        });
      };

      await readEntries(entry);
      return files;
    },
    [],
  );

  // Main file processing function
  const processFiles = useCallback(
    async (dataTransfer: DataTransfer) => {
      const attachments: Array<{
        filename: string;
        data: Uint8Array;
        type?: string;
      }> = [];
      const errors: Array<{ filename: string; error: string }> = [];
      const allFiles: File[] = [];

      // Extract files from DataTransfer
      if (dataTransfer.items) {
        for (const item of dataTransfer.items) {
          if (item.kind === "file") {
            const entry = item.webkitGetAsEntry?.();
            if (entry?.isFile) {
              const file = item.getAsFile();
              if (file) allFiles.push(file);
            } else if (entry?.isDirectory) {
              const dirEntry = entry as FileSystemDirectoryEntry;
              const dirFiles = await readDirectory(dirEntry);
              allFiles.push(...dirFiles);
            }
          }
        }
      } else if (dataTransfer.files) {
        allFiles.push(...Array.from(dataTransfer.files));
      }

      // First pass: Check file sizes and types
      const validFiles: File[] = [];

      for (const file of allFiles) {
        const fileExtension = file.name.toLowerCase().split(".").pop();

        // Custom validation first
        if (validateFile) {
          const validation = validateFile(file);
          if (!validation.valid) {
            errors.push({
              filename: file.name,
              error: validation.error || "File validation failed",
            });
            continue;
          }
        }

        // Default validation
        if (!fileExtension) {
          errors.push({
            filename: file.name,
            error: "File type not supported",
          });
        } else if (
          IMAGE_EXTENSIONS.includes(fileExtension) &&
          !hasVisionCapability
        ) {
          errors.push({
            filename: file.name,
            error: "This model does not support images",
          });
        } else if (!ALLOWED_EXTENSIONS.includes(fileExtension)) {
          errors.push({
            filename: file.name,
            error: "File type not supported",
          });
        } else if (file.size > MAX_FILE_SIZE) {
          errors.push({
            filename: file.name,
            error: "File too large",
          });
        } else {
          validFiles.push(file);
        }
      }

      // Second pass: Process only valid files
      for (const file of validFiles) {
        try {
          const fileBytes = await readFileAsBytes(file);
          attachments.push({
            filename: file.name,
            data: fileBytes,
            type: file.type || undefined,
          });
        } catch (error) {
          console.error(`Error reading file ${file.name}:`, error);
          errors.push({
            filename: file.name,
            error: "Error reading file",
          });
        }
      }

      // Send processed files and errors back to parent
      if (attachments.length > 0 || errors.length > 0) {
        onFilesAdded(attachments, errors);
      }
    },
    [
      readFileAsBytes,
      readDirectory,
      selectedModel,
      hasVisionCapability,
      ALLOWED_EXTENSIONS,
      MAX_FILE_SIZE,
      validateFile,
      onFilesAdded,
    ],
  );

  // Drag event handlers
  const handleDragEnter = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      dragCounter.current++;

      if (hasFiles(e.dataTransfer)) {
        setIsDragging(true);
      }
    },
    [hasFiles],
  );

  // Paste event handler
  const handlePaste = useCallback(
    async (e: ClipboardEvent) => {
      // Check if clipboard contains files
      if (e.clipboardData && e.clipboardData.files.length > 0) {
        // Check if there's text data in the clipboard
        // Only process files if there's no text data
        const hasTextData =
          e.clipboardData.types.includes("text/plain") &&
          e.clipboardData.getData("text/plain").trim().length > 0;

        if (hasTextData) {
          return;
        }

        e.preventDefault();

        // Create a synthetic DataTransfer object for our processFiles function
        const items = Array.from(e.clipboardData.items);
        const syntheticDataTransfer = {
          files: e.clipboardData.files,
          items: {
            ...items,
            length: items.length,
            add: () => null,
            clear: () => {},
            remove: () => null,
            [Symbol.iterator]: () => items[Symbol.iterator](),
          } as DataTransferItemList,
          types: e.clipboardData.types,
          getData: (format: string) => e.clipboardData!.getData(format),
        } as DataTransfer;

        await processFiles(syntheticDataTransfer);
      }
    },
    [processFiles],
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current--;

    // Only hide overlay when leaving all elements in the component tree
    if (dragCounter.current === 0) {
      setIsDragging(false);
    }
  }, []);

  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      dragCounter.current = 0;
      setIsDragging(false);

      if (hasFiles(e.dataTransfer) && e.dataTransfer) {
        await processFiles(e.dataTransfer);
      }
    },
    [hasFiles, processFiles],
  );

  // Set up paste event listener
  useEffect(() => {
    document.addEventListener("paste", handlePaste);
    return () => {
      document.removeEventListener("paste", handlePaste);
    };
  }, [handlePaste]);

  return (
    <div
      className="relative"
      onDragEnter={handleDragEnter}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {children}

      {/* Drop zone overlay */}
      {isDragging && (
        <div className="absolute inset-0 z-[9999] pointer-events-none">
          <div className="absolute inset-0 bg-neutral-500/5 dark:bg-neutral-400/10 transition-opacity duration-200" />
          <div className="absolute inset-0 bg-neutral-50 bg-opacity-90 dark:bg-neutral-900 dark:bg-opacity-30 flex items-center justify-center">
            <div className="bg-white/90 dark:bg-neutral-900/90 backdrop-blur-xl rounded-2xl p-12 mx-4 max-w-sm text-center border border-neutral-200/50 dark:border-neutral-700/50 shadow-2xl">
              <DocumentPlusIcon className="w-8 h-8 mx-auto mb-2 text-black dark:text-white" />
              <p className="text-neutral-500 dark:text-neutral-400 text-sm font-medium leading-relaxed">
                Drop files here or paste from clipboard to add them to your
                message
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
