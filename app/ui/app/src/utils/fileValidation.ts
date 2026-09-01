import { Model } from "@/gotypes";
// Shared file validation logic used by both FileUpload and native dialog selection

export const TEXT_FILE_EXTENSIONS = [
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

export const IMAGE_EXTENSIONS = ["png", "jpg", "jpeg", "webp"];

export interface FileValidationOptions {
  maxFileSize?: number; // in MB
  allowedExtensions?: string[];
  hasVisionCapability?: boolean;
  selectedModel?: Model | null;
  customValidator?: (file: File) => { valid: boolean; error?: string };
}

export interface ValidationResult {
  valid: boolean;
  error?: string;
}

export function validateFile(
  file: File,
  options: FileValidationOptions = {},
): ValidationResult {
  const {
    maxFileSize = 10,
    allowedExtensions = [...TEXT_FILE_EXTENSIONS, ...IMAGE_EXTENSIONS],
    customValidator,
  } = options;

  const MAX_FILE_SIZE = maxFileSize * 1024 * 1024; // Convert MB to bytes
  const fileExtension = file.name.toLowerCase().split(".").pop();

  // Custom validation first
  if (customValidator) {
    const customResult = customValidator(file);
    if (!customResult.valid) {
      return customResult;
    }
  }

  // File extension validation
  if (!fileExtension || !allowedExtensions.includes(fileExtension)) {
    return { valid: false, error: "File type not supported" };
  }

  // File size validation
  if (file.size > MAX_FILE_SIZE) {
    return { valid: false, error: "File too large" };
  }

  return { valid: true };
}

// Helper function to read file as Uint8Array
export function readFileAsBytes(file: File): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const arrayBuffer = reader.result as ArrayBuffer;
      resolve(new Uint8Array(arrayBuffer));
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsArrayBuffer(file);
  });
}

// Re-encode image bytes through canvas to guarantee a format llama.cpp can decode.
// macOS clipboard provides images as TIFF bytes mislabeled with type "image/png";
// stb_image (used by llama.cpp) does not support TIFF. Re-encoding via
// createImageBitmap + canvas.toBlob("image/png") strips any problematic container
// and always produces valid PNG data.
async function ensureValidImageBytes(
  data: Uint8Array,
  mimeType: string,
): Promise<Uint8Array> {
  if (!mimeType?.startsWith("image/")) return data;

  try {
    const blob = new Blob([data], { type: mimeType });
    const imageBitmap = await createImageBitmap(blob);

    const canvas = document.createElement("canvas");
    canvas.width = imageBitmap.width;
    canvas.height = imageBitmap.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return data;
    ctx.drawImage(imageBitmap, 0, 0);
    imageBitmap.close();

    const pngBlob = await new Promise<Blob>((resolve) => {
      canvas.toBlob((b) => resolve(b!), "image/png");
    });

    const arrayBuffer = await pngBlob.arrayBuffer();
    return new Uint8Array(arrayBuffer);
  } catch (e) {
    console.warn("Failed to re-encode image, sending original bytes:", e);
    return data;
  }
}

// Process multiple files with validation
export async function processFiles(
  files: File[],
  options: FileValidationOptions = {},
): Promise<{
  validFiles: Array<{ filename: string; data: Uint8Array; type?: string }>;
  errors: Array<{ filename: string; error: string }>;
}> {
  const validFiles: Array<{
    filename: string;
    data: Uint8Array;
    type?: string;
  }> = [];
  const errors: Array<{ filename: string; error: string }> = [];

  for (const file of files) {
    const validation = validateFile(file, options);

    if (!validation.valid) {
      errors.push({
        filename: file.name,
        error: validation.error || "File validation failed",
      });
      continue;
    }

    try {
      const fileBytes = await readFileAsBytes(file);
      const normalizedBytes = await ensureValidImageBytes(fileBytes, file.type);
      validFiles.push({
        filename: file.name,
        data: normalizedBytes,
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

  return { validFiles, errors };
}
