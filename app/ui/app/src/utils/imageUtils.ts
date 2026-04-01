export function isImageFile(filename: string): boolean {
  const extension = filename.toLowerCase().split(".").pop();
  return ["png", "jpg", "jpeg", "gif", "webp"].includes(extension || "");
}
