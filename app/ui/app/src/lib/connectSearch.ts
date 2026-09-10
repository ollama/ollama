// Search params the onboarding screen uses to hand an intent to the Apps page.
// `connect` starts a desktop connection; `highlight` scrolls to one row.
export interface ConnectSearch {
  connect?: boolean | "chatgpt";
  highlight?: string;
}

export function parseConnectSearch(
  search: Record<string, unknown>,
): ConnectSearch {
  const result: ConnectSearch = {};
  if (search.connect === true || search.connect === "true") {
    result.connect = true;
  } else if (search.connect === "chatgpt") {
    result.connect = "chatgpt";
  }
  if (typeof search.highlight === "string" && search.highlight !== "") {
    result.highlight = search.highlight;
  }
  return result;
}
