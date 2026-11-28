import { useEffect, useState } from "react";
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

function buildImageUrl(filename: string, data: ImageData["data"]): string {
  if (!isImageFile(filename)) return "";

  try {
    // Determine MIME type from file extension
    const extension = filename.toLowerCase().split(".").pop();
    let mimeType = "application/octet-stream";

    switch (extension) {
      case "png":
        mimeType = "image/png";
        break;
      case "jpg":
      case "jpeg":
        mimeType = "image/jpeg";
        break;
      case "gif":
        mimeType = "image/gif";
        break;
      case "webp":
        mimeType = "image/webp";
        break;
    }

    // Convert to Uint8Array if needed
    let bytes: Uint8Array;
    if (data instanceof Uint8Array) {
      bytes = data;
    } else if (Array.isArray(data)) {
      bytes = new Uint8Array(data);
    } else if (typeof data === "string") {
      // Convert base64 string to Uint8Array
      const binaryString = atob(data);
      bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
    } else {
      console.error("Invalid data format for:", filename, typeof data);
      return "";
    }

    const blob = new Blob([bytes], { type: mimeType });
    return URL.createObjectURL(blob);
  } catch (error) {
    console.error(
      "Error converting file data to URL for",
      filename,
      ":",
      error,
    );
    return "";
  }
}

export function ImageThumbnail({
  image,
  className = "w-16 h-16 object-cover rounded-md select-none",
  alt,
  onError,
}: ImageThumbnailProps) {
  const [imageLoadError, setImageLoadError] = useState(false);
  const [imageUrl, setImageUrl] = useState("");

  // Create the blob URL in an effect keyed on the image fields rather than
  // object identity: callers may build a fresh `image` object on every
  // render, and recreating the URL each render forces the <img> to re-decode
  // on each keystroke (#17540). An effect (unlike useMemo) also revokes and
  // recreates the URL correctly under StrictMode's remount cycle.
  useEffect(() => {
    const url = buildImageUrl(image.filename, image.data);
    setImageUrl(url);
    setImageLoadError(false);
    return () => {
      if (url) {
        URL.revokeObjectURL(url);
      }
    };
  }, [image.filename, image.data]);

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
