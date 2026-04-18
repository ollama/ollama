import { useMemo, useState } from "react";
import { isImageFile } from "@/utils/imageUtils";

export interface ImageData {
  filename: string;
  data: Uint8Array | number[] | string;
  type?: string;
}

interface ImageThumbnailProps {
  image: ImageData;
  className?: string;
  alt?: string;
  onError?: () => void;
}

const MIME_TYPES: Record<string, string> = {
  png: "image/png",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  gif: "image/gif",
  webp: "image/webp",
};

function toBase64(data: Uint8Array | number[] | string): string {
  if (typeof data === "string") return data;
  const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
  let binary = "";
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

export function ImageThumbnail({
  image,
  className = "w-16 h-16 object-cover rounded-md select-none",
  alt,
  onError,
}: ImageThumbnailProps) {
  const [imageLoadError, setImageLoadError] = useState(false);

  // Data URL (not blob URL) — stable across re-renders, no revocation needed
  const dataLength =
    typeof image.data === "string" ? image.data.length : image.data?.length;

  const imageUrl = useMemo(() => {
    if (!isImageFile(image.filename) || !image.data || dataLength === 0)
      return "";

    try {
      const ext = image.filename.toLowerCase().split(".").pop() || "";
      const mime = MIME_TYPES[ext] || "application/octet-stream";
      return `data:${mime};base64,${toBase64(image.data)}`;
    } catch (error) {
      console.error("Error converting image data for", image.filename, error);
      return "";
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- stable on filename + data length
  }, [image.filename, dataLength]);

  if (!isImageFile(image.filename) || !imageUrl) {
    return null;
  }

  if (imageLoadError) {
    return (
      <div
        className={`flex items-center justify-center bg-neutral-50 dark:bg-neutral-600/50 rounded-md ${className}`}
      >
        <svg
          className="w-4 h-4 text-neutral-400 dark:text-neutral-500"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
          />
        </svg>
      </div>
    );
  }

  return (
    <img
      src={imageUrl}
      alt={alt || image.filename}
      className={className}
      onError={() => {
        setImageLoadError(true);
        onError?.();
      }}
    />
  );
}
