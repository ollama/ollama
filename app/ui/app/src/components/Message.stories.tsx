import type { Meta, StoryObj } from "@storybook/react-vite";
import Message from "./Message";
import { Message as MessageType, ToolCall, ToolFunction } from "@/gotypes";

const meta = {
  title: "Components/Message",
  component: Message,
  parameters: {
    layout: "padded",
  },
  tags: ["autodocs"],
  argTypes: {
    message: {
      description: "The message object to display",
    },
  },
} satisfies Meta<typeof Message>;

export default meta;
type Story = StoryObj<typeof meta>;

// Helper function to create a message
const createMessage = (overrides: Partial<MessageType>): MessageType => {
  const now = new Date();
  return new MessageType({
    role: "user",
    content: "Hello, world!",
    thinking: "",
    stream: false,
    created_at: now.toISOString(),
    updated_at: now.toISOString(),
    ...overrides,
  });
};

// User Messages
export const UserMessage: Story = {
  args: {
    message: createMessage({
      role: "user",
      content: "Can you help me understand how React hooks work?",
    }),
    isStreaming: false,
  },
};

export const UserMessageWithMarkdown: Story = {
  args: {
    message: createMessage({
      role: "user",
      content:
        "Here's my code:\n```javascript\nconst [count, setCount] = useState(0);\n```\nWhy isn't it working?",
    }),
    isStreaming: false,
  },
};

// Assistant Messages
export const AssistantMessage: Story = {
  args: {
    message: createMessage({
      role: "assistant",
      content:
        "I'd be happy to help you understand React hooks! React hooks are functions that let you use state and other React features in functional components.",
    }),
    isStreaming: false,
  },
};

export const AssistantMessageWithCodeBlock: Story = {
  args: {
    message: createMessage({
      role: "assistant",
      content: `Here's an example of using the useState hook:

\`\`\`javascript
import React, { useState } from 'react';

function Counter() {
  const [count, setCount] = useState(0);
  
  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>
        Increment
      </button>
    </div>
  );
}
\`\`\`

This creates a simple counter component that tracks its state.`,
    }),
    isStreaming: false,
  },
};

export const AssistantMessageWithThinking: Story = {
  args: {
    message: createMessage({
      role: "assistant",
      content:
        "Based on my analysis, the issue with your code is that you need to import useState from React.",
      thinking:
        "The user is trying to use useState but hasn't imported it. This is a common mistake for beginners. I should provide a clear explanation and show the correct import statement.",
      thinkingTimeStart: new Date(Date.now() - 3000),
      thinkingTimeEnd: new Date(Date.now() - 1000),
    }),
    isStreaming: false,
  },
};

export const AssistantMessageThinkingOnly: Story = {
  args: {
    message: createMessage({
      role: "assistant",
      content: "",
      thinking:
        "Processing the user's request and analyzing the code structure. This might take a moment while I consider the best approach...",
      thinkingTimeStart: new Date(Date.now() - 2000),
    }),
    isStreaming: false,
  },
};

// Tool Messages
export const ToolMessage: Story = {
  args: {
    message: createMessage({
      role: "tool",
      content: `{
  "status": "success",
  "data": {
    "temperature": 72,
    "humidity": 45,
    "location": "San Francisco"
  }
}`,
    }),
    isStreaming: false,
  },
};

// Messages with Tool Calls
export const AssistantWithToolCall: Story = {
  args: {
    message: createMessage({
      role: "assistant",
      content: "I'll check the current weather for you.",
      tool_calls: [
        new ToolCall({
          type: "function",
          function: new ToolFunction({
            name: "get_weather",
            arguments: JSON.stringify({
              location: "San Francisco",
              units: "fahrenheit",
            }),
            result: {
              temperature: 72,
              humidity: 45,
              conditions: "sunny",
            },
          }),
        }),
      ],
    }),
    isStreaming: false,
  },
};

export const AssistantWithMultipleToolCalls: Story = {
  args: {
    message: createMessage({
      role: "assistant",
      content: "Let me gather some information for you.",
      tool_calls: [
        new ToolCall({
          type: "function",
          function: new ToolFunction({
            name: "search_web",
            arguments: JSON.stringify({
              query: "React hooks best practices",
              limit: 5,
            }),
          }),
        }),
        new ToolCall({
          type: "function",
          function: new ToolFunction({
            name: "read_documentation",
            arguments: JSON.stringify({
              url: "https://react.dev/reference/react/hooks",
              section: "useState",
            }),
            result:
              "useState is a React Hook that lets you add a state variable to your component.",
          }),
        }),
      ],
    }),
    isStreaming: false,
  },
};

// Complex Message with Everything
export const ComplexAssistantMessage: Story = {
  args: {
    message: createMessage({
      role: "assistant",
      content: `## React Hooks Best Practices

Based on my research, here are the key best practices:

1. **Only call hooks at the top level** - Don't call hooks inside loops, conditions, or nested functions
2. **Only call hooks from React functions** - Either from React function components or custom hooks
3. **Use the ESLint plugin** - Install \`eslint-plugin-react-hooks\` to enforce these rules

### Example of correct usage:

\`\`\`javascript
function MyComponent() {
  // ✅ Good - hooks at the top level
  const [count, setCount] = useState(0);
  const theme = useContext(ThemeContext);
  
  if (count > 5) {
    // ❌ Bad - hook inside condition
    // const [error, setError] = useState(null);
  }
  
  return <div>{count}</div>;
}
\`\`\``,
      thinking:
        "The user needs a comprehensive guide on React hooks best practices. I should cover the rules of hooks, provide examples, and maybe include some tool calls to fetch the latest documentation.",
      tool_calls: [
        new ToolCall({
          type: "function",
          function: new ToolFunction({
            name: "fetch_documentation",
            arguments: JSON.stringify({
              topic: "react-hooks-rules",
            }),
            result: "Successfully fetched React hooks documentation",
          }),
        }),
      ],
      thinkingTimeStart: new Date(Date.now() - 5000),
      thinkingTimeEnd: new Date(Date.now() - 3000),
    }),
    isStreaming: false,
  },
};

// Assistant Message with Raw HTML
export const AssistantMessageWithHTML: Story = {
  args: {
    isStreaming: false,
    message: createMessage({
      role: "assistant",
      content: `Here are some HTML examples and how they render:

## Basic HTML Elements

<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">
  <p>This is a <strong>paragraph</strong> inside a <em>styled div</em>.</p>
</div>

<blockquote>
  This is a blockquote with <code>inline code</code> and <a href="#">a link</a>.
</blockquote>

## Lists and Tables

<ul>
  <li>First item with <b>bold text</b></li>
  <li>Second item with <i>italic text</i></li>
  <li>Third item with <mark>highlighted text</mark></li>
</ul>

<table>
  <thead>
    <tr>
      <th>Header 1</th>
      <th>Header 2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Cell 1</td>
      <td>Cell 2</td>
    </tr>
    <tr>
      <td>Cell 3</td>
      <td>Cell 4</td>
    </tr>
  </tbody>
</table>

## Forms and Inputs

<details>
  <summary>Click to expand</summary>
  <p>Hidden content here with <span style="color: red;">colored text</span>.</p>
</details>

<pre><code class="language-html">&lt;button&gt;This is HTML code&lt;/button&gt;</code></pre>

## Potentially Dangerous HTML

<script>console.log('This script should not execute');</script>
<iframe src="https://example.com">This iframe should not render</iframe>
<img src="x" onerror="alert('XSS')" />
<a href="javascript:alert('XSS')">Dangerous link</a>

## Special Characters

&lt;div&gt; &amp; &quot;quotes&quot; &apos;apostrophes&apos; &copy; &reg; &trade;

## Inline Styles

<p style="font-size: 20px; color: blue;">Large blue text</p>
<span style="background: yellow; padding: 2px 4px;">Highlighted span</span>
<div style="border: 2px solid green; margin: 10px 0; padding: 10px;">
  Box with green border
</div>`,
    }),
  },
};

// Long Content Message
export const LongContentMessage: Story = {
  args: {
    message: createMessage({
      role: "assistant",
      content: `# Comprehensive Guide to React Hooks

React Hooks revolutionized how we write React components by allowing us to use state and other React features without writing a class. Let's dive deep into understanding hooks.

## What are React Hooks?

Hooks are functions that let you "hook into" React state and lifecycle features from function components. They don't work inside classes — they let you use React without classes.

## Core Hooks

### useState

The State Hook lets you add React state to function components.

\`\`\`javascript
const [state, setState] = useState(initialState);
\`\`\`

### useEffect

The Effect Hook lets you perform side effects in function components:

\`\`\`javascript
useEffect(() => {
  // Side effect logic here
  return () => {
    // Cleanup logic here
  };
}, [dependencies]);
\`\`\`

### useContext

Accepts a context object and returns the current context value:

\`\`\`javascript
const value = useContext(MyContext);
\`\`\`

## Additional Hooks

- **useReducer**: An alternative to useState for complex state logic
- **useCallback**: Returns a memoized callback
- **useMemo**: Returns a memoized value
- **useRef**: Returns a mutable ref object
- **useImperativeHandle**: Customizes the instance value exposed to parent components
- **useLayoutEffect**: Similar to useEffect, but fires synchronously
- **useDebugValue**: Displays a label for custom hooks in React DevTools

## Custom Hooks

You can create your own hooks to reuse stateful logic between components:

\`\`\`javascript
function useCounter(initialValue = 0) {
  const [count, setCount] = useState(initialValue);
  
  const increment = useCallback(() => {
    setCount(c => c + 1);
  }, []);
  
  const decrement = useCallback(() => {
    setCount(c => c - 1);
  }, []);
  
  return { count, increment, decrement };
}
\`\`\`

## Best Practices and Rules

1. **Only Call Hooks at the Top Level**
2. **Only Call Hooks from React Functions**
3. **Use the ESLint Plugin**
4. **Keep Effects Clean**
5. **Optimize with useMemo and useCallback**

This is just the beginning of what you can do with React Hooks!`,
    }),
    isStreaming: false,
  },
};
