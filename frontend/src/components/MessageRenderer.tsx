import React from 'react';
import './MessageRenderer.css';

interface MessageRendererProps {
  content: string;
  codeMode: boolean;
}

const MessageRenderer: React.FC<MessageRendererProps> = ({ content, codeMode }) => {
  if (!codeMode) {
    return <div className="message-content">{content}</div>;
  }

  const parseMarkdown = (text: string): React.ReactNode[] => {
    const parts: React.ReactNode[] = [];
    const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g;
    let lastIndex = 0;
    let match;
    let key = 0;

    while ((match = codeBlockRegex.exec(text)) !== null) {
      if (match.index > lastIndex) {
        const beforeText = text.substring(lastIndex, match.index);
        parts.push(
          <div key={`text-${key++}`} className="message-text">
            {formatInlineCode(beforeText)}
          </div>
        );
      }

      const language = match[1] || 'text';
      const code = match[2];
      parts.push(
        <CodeBlock key={`code-${key++}`} language={language} code={code} />
      );

      lastIndex = match.index + match[0].length;
    }

    if (lastIndex < text.length) {
      const remainingText = text.substring(lastIndex);
      parts.push(
        <div key={`text-${key++}`} className="message-text">
          {formatInlineCode(remainingText)}
        </div>
      );
    }

    return parts.length > 0 ? parts : [<div key="empty" className="message-text">{text}</div>];
  };

  const formatInlineCode = (text: string): React.ReactNode[] => {
    const parts: React.ReactNode[] = [];
    const inlineCodeRegex = /`([^`]+)`/g;
    let lastIndex = 0;
    let match;
    let key = 0;

    while ((match = inlineCodeRegex.exec(text)) !== null) {
      if (match.index > lastIndex) {
        parts.push(text.substring(lastIndex, match.index));
      }
      parts.push(
        <code key={`inline-${key++}`} className="inline-code">
          {match[1]}
        </code>
      );
      lastIndex = match.index + match[0].length;
    }

    if (lastIndex < text.length) {
      parts.push(text.substring(lastIndex));
    }

    return parts.length > 0 ? parts : [text];
  };

  return <div className="markdown-content">{parseMarkdown(content)}</div>;
};

interface CodeBlockProps {
  language: string;
  code: string;
}

const CodeBlock: React.FC<CodeBlockProps> = ({ language, code }) => {
  const lines = code.split('\n').filter((line, idx, arr) => {
    if (idx === arr.length - 1 && line.trim() === '') return false;
    return true;
  });

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
  };

  return (
    <div className="code-block-wrapper">
      <div className="code-block-header">
        <span className="code-language">{language}</span>
        <button className="copy-button" onClick={handleCopy} title="Copy code">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path
              d="M5.75 4.75H10.25V1.75H5.75V4.75ZM4.5 1.75C4.5 1.05964 5.05964 0.5 5.75 0.5H10.25C10.9404 0.5 11.5 1.05964 11.5 1.75V4.75H13.25C14.0784 4.75 14.75 5.42157 14.75 6.25V13.25C14.75 14.0784 14.0784 14.75 13.25 14.75H2.75C1.92157 14.75 1.25 14.0784 1.25 13.25V6.25C1.25 5.42157 1.92157 4.75 2.75 4.75H4.5V1.75Z"
              fill="currentColor"
            />
          </svg>
        </button>
      </div>
      <div className="code-block-content">
        <div className="line-numbers">
          {lines.map((_, idx) => (
            <div key={idx} className="line-number">
              {idx + 1}
            </div>
          ))}
        </div>
        <pre className="code-pre">
          <code className={`language-${language}`}>
            {lines.map((line, idx) => (
              <div key={idx} className="code-line">
                {line || '\n'}
              </div>
            ))}
          </code>
        </pre>
      </div>
    </div>
  );
};

export default MessageRenderer;
