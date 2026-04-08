export async function copyTextToClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (clipboardError) {
    console.error(
      "Clipboard API failed, falling back to execCommand",
      clipboardError,
    );
  }

  try {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "true");
    textarea.style.position = "fixed";
    textarea.style.left = "-9999px";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();

    const copied = document.execCommand("copy");
    document.body.removeChild(textarea);
    return copied;
  } catch (fallbackError) {
    console.error("Fallback copy failed", fallbackError);
    return false;
  }
}
