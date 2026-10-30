import { useMemo, useEffect, useState } from "react";
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

export function ImageThumbnail({
  image,
  className = "w-16 h-16 object-cover rounded-md select-none",
  alt,
  onError,
}: ImageThumbnailProps) {
  const [imageLoadError, setImageLoadError] = useState(false);

  const imageUrl = useMemo(() => {
    if (!isImageFile(image.filename)) return "";

    try {
      // Determine MIME type from file extension
      const extension = image.filename.toLowerCase().split(".").pop();
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
      if (image.data instanceof Uint8Array) {
        bytes = image.data;
      } else if (Array.isArray(image.data)) {
        bytes = new Uint8Array(image.data);
      } else if (typeof image.data === "string") {
        // Convert base64 string to Uint8Array
        const binaryString = atob(image.data);
        bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }
      } else {
        console.error(
          "Invalid data format for:",
          image.filename,
          typeof image.data,
        );
        return "";
      }

      const blob = new Blob([bytes], { type: mimeType });
      return URL.createObjectURL(blob);
    } catch (error) {
      console.error(
        "Error converting file data to URL for",
        image.filename,
        ":",
        error,
      );
      return "";
    }
  }, [image]);

  // Cleanup blob URL on unmount and reset error state when image changes
  useEffect(() => {
    setImageLoadError(false);
    return () => {
      if (imageUrl) {
        URL.revokeObjectURL(imageUrl);
      }
    };
  }, [imageUrl]);

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
