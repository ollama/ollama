# Documentation Lookup Strategy

## When to Use Context7

Use Context7's `resolve-library-id` and `query-docs` tools when:

- The question involves an **external library/framework API** (FastAPI, React, Next.js, Ollama, etc.)
- The question is **version-specific** (e.g., "Next.js 15 `after()` function", "React 19 hooks")
- You need **migration guides** or **changelog information**
- The user explicitly requests: "use context7", "check docs", "use official docs", "fetch docs"
- You're **generating integration code** against external SDKs and need current API references
- The question involves **deprecated methods** or **breaking changes**

## When NOT to Use Context7

Skip Context7 when:

- The question is about **this repository's own code** — use codebase search and linked GitHub session context instead
- The answer is in **files already open** or in the current conversation context
- It's a **general reasoning** question, architecture opinion, or code review
- It's a **rewrite/refactor/summarize** task with no external API dependencies
- The question is for **small local models** (like llama3.2:1b) where extra tokens often hurt more than help
- You're working with the **gateway's GitHub repo integration** — that takes priority for this-repo questions

## Best Practices

1. **Resolve library ID once** — Use `resolve-library-id` to find the correct library, then cache the result
2. **Focus the query** — Pass a specific topic/question to `query-docs` rather than broad requests
3. **Cap token usage** — Don't fetch docs if the context window is already full
4. **Combine strategically** — Use codebase search for local code + Context7 for external APIs
5. **Check context first** — If the answer is already in open files or recent conversation, skip the lookup

## Priority Order

When deciding what context to use:

1. **GitHub session context** — If a repo is linked and the question is about that codebase
2. **Open files/codebase search** — For this project's implementation details
3. **Context7** — For external library documentation and version-specific APIs
4. **Model knowledge only** — For general programming concepts and reasoning

## Example Triggers

**Use Context7:**
- "How do I use Ollama's streaming API in FastAPI?"
- "What's new in React 19 concurrent features?"
- "Context7: show me Next.js 15 server actions"
- "How to configure Redis connection pooling?"

**Don't use Context7:**
- "Explain how our gateway session management works" (use codebase search)
- "Refactor the Chat component for better performance" (local code)
- "What's the difference between async/await and promises?" (general knowledge)
- "Review this code for bugs" (code review, no external API needed)
