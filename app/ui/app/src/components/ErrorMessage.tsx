import type { ErrorEvent } from "@/gotypes";

interface ErrorMessageProps {
  error: ErrorEvent;
}

const renderWithLinks = (text: string) => {
  const urlRegex =
    /(https?:\/\/(?!127\.|localhost|0\.0\.0\.0|10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[0-1])\.)[^\s]+)/g;

  const urls: string[] = [];
  let match;
  while ((match = urlRegex.exec(text)) !== null) {
    urls.push(match[0]);
  }

  urlRegex.lastIndex = 0;

  // Split text by URLs
  // for example if the text is "connection failed. Try visiting https://ollama.com or check https://ollama.com/doc for help"
  // then the parsed parts will be ["connection failed. Try visiting ", "https://ollama.com", " or check ", "https://ollama.com/doc", " for help"]
  const parts = text.split(urlRegex);

  return parts.map((part, index) => {
    if (urls.includes(part)) {
      return (
        <a
          key={index}
          href={part}
          target="_blank"
          rel="noopener noreferrer"
          className="underline break-all"
          onClick={(e) => {
            e.stopPropagation();
          }}
        >
          {part}
        </a>
      );
    }
    return part;
  });
};

export const ErrorMessage = ({ error }: ErrorMessageProps) => {
  return (
    <div className="flex flex-col w-full text-neutral-800 dark:text-neutral-200 transition-colors mb-8">
      <div className="flex items-center self-start relative">
        {/* Circled X icon */}
        <div className="flex-shrink-0 w-4 h-4 rounded-full border border-black dark:border-white flex items-center justify-center mr-3">
          <svg
            className="w-2.5 h-2.5 fill-current text-black dark:text-white"
            viewBox="0 0 20 20"
            fill="none"
          >
            <path
              fillRule="evenodd"
              d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
              clipRule="evenodd"
            />
          </svg>
        </div>
        <h3>Error</h3>
      </div>

      <div className="flex items-start ml-[1.8rem] mt-2">
        <div className="text-sm text-neutral-500 dark:text-neutral-500 opacity-75 flex-1">
          {renderWithLinks(error.error)}
        </div>
      </div>
    </div>
  );
};
