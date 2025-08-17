import React from 'react'

// Lightweight markdown renderer supporting:
// - Headings (# .. ######)
// - Inline code `code`
// - Code fences ```lang ... ``` with copy button
// - Bold **text** and italics *text* / _text_
// - Unordered lists (-, *, +) and ordered lists (1.)
// - Paragraphs & line breaks
// Future (deferred): tables, blockquotes, links autolinking.

interface MarkdownProps { content: string }

interface BlockBase { type: string; key: string }
interface ParagraphBlock extends BlockBase { type: 'paragraph'; parts: InlinePart[] }
interface HeadingBlock extends BlockBase { type: 'heading'; level: number; parts: InlinePart[] }
interface CodeBlock extends BlockBase { type: 'code'; language?: string; code: string }
interface ListBlock extends BlockBase { type: 'list'; ordered: boolean; items: InlinePart[][] }

type Block = ParagraphBlock | HeadingBlock | CodeBlock | ListBlock

type InlinePart = 
  | { t: 'text'; v: string }
  | { t: 'code'; v: string }
  | { t: 'strong'; v: InlinePart[] }
  | { t: 'em'; v: InlinePart[] }

// Basic inline tokenizer (nested strong/em limited depth)
function parseInline(src: string): InlinePart[] {
  const out: InlinePart[] = []
  let i = 0
  function pushText(t: string) { if (!t) return; out.push({ t: 'text', v: t }) }
  while (i < src.length) {
    // Inline code
    if (src[i] === '`') {
      const end = src.indexOf('`', i+1)
      if (end > i+1) {
        out.push({ t: 'code', v: src.slice(i+1, end) })
        i = end + 1
        continue
      }
    }
    // Bold **
    if (src[i] === '*' && src[i+1] === '*') {
      const end = src.indexOf('**', i+2)
      if (end > i+2) {
        const inner = parseInline(src.slice(i+2, end))
        out.push({ t: 'strong', v: inner })
        i = end + 2
        continue
      }
    }
    // Italic *
    if (src[i] === '*') {
      const end = src.indexOf('*', i+1)
      if (end > i+1) {
        const inner = parseInline(src.slice(i+1, end))
        out.push({ t: 'em', v: inner })
        i = end + 1
        continue
      }
    }
    // Italic _
    if (src[i] === '_') {
      const end = src.indexOf('_', i+1)
      if (end > i+1) {
        const inner = parseInline(src.slice(i+1, end))
        out.push({ t: 'em', v: inner })
        i = end + 1
        continue
      }
    }
    // Plain text accumulation
    let j = i
    while (j < src.length && !['`','*','_'].includes(src[j])) j++
    pushText(src.slice(i, j))
    i = j
  }
  return out
}

function parseBlocks(content: string): Block[] {
  const lines = content.replace(/\r\n?/g,'\n').split('\n')
  const blocks: Block[] = []
  let i = 0
  let listBuffer: { ordered: boolean; items: string[] } | null = null

  function flushParagraphBuffer(paras: string[]) {
    if (!paras.length) return
    const text = paras.join('\n')
    blocks.push({ type: 'paragraph', key: 'p'+blocks.length, parts: parseInline(text) })
    paras.length = 0
  }
  function flushList() {
    if (!listBuffer) return
    const itemsParts = listBuffer.items.map(item => parseInline(item))
    blocks.push({ type: 'list', key: 'l'+blocks.length, ordered: listBuffer.ordered, items: itemsParts })
    listBuffer = null
  }

  const paragraphAccum: string[] = []

  while (i < lines.length) {
    const line = lines[i]

    // Code fence start
    const fenceMatch = line.match(/^```(.*)$/)
    if (fenceMatch) {
      flushParagraphBuffer(paragraphAccum)
      flushList()
      const lang = (fenceMatch[1] || '').trim() || undefined
      i++
      const codeLines: string[] = []
      while (i < lines.length && !/^```/.test(lines[i])) { codeLines.push(lines[i]); i++ }
      if (i < lines.length && /^```/.test(lines[i])) i++ // consume closing
      blocks.push({ type: 'code', key: 'c'+blocks.length, language: lang, code: codeLines.join('\n') })
      continue
    }

    // Heading
    const headingMatch = line.match(/^(#{1,6})\s+(.*)$/)
    if (headingMatch) {
      flushParagraphBuffer(paragraphAccum)
      flushList()
      const level = headingMatch[1].length
      blocks.push({ type: 'heading', key: 'h'+blocks.length, level, parts: parseInline(headingMatch[2].trim()) })
      i++
      continue
    }

    // List item
    const ulMatch = line.match(/^\s*([-*+])\s+(.*)$/)
    const olMatch = line.match(/^\s*(\d+)\.\s+(.*)$/)
    if (ulMatch || olMatch) {
      flushParagraphBuffer(paragraphAccum)
      const ordered = !!olMatch
      const itemText = (ulMatch ? ulMatch[2] : olMatch![2]).trim()
      if (listBuffer && listBuffer.ordered === ordered) {
        listBuffer.items.push(itemText)
      } else {
        flushList()
        listBuffer = { ordered, items: [itemText] }
      }
      i++
      continue
    }

    // Blank line flush states
    if (/^\s*$/.test(line)) {
      flushParagraphBuffer(paragraphAccum)
      flushList()
      i++
      continue
    }

    // Accumulate paragraph
    paragraphAccum.push(line)
    i++
  }
  flushParagraphBuffer(paragraphAccum)
  flushList()
  return blocks
}

const InlineRenderer: React.FC<{ parts: InlinePart[] }> = ({ parts }) => {
  return <>{parts.map((p, idx) => {
    switch (p.t) {
      case 'text': return <span key={idx}>{p.v}</span>
      case 'code': return <code key={idx} className='px-1 py-[2px] rounded bg-[#1e1e1e] border border-[#262626] font-mono text-[12px]'>{p.v}</code>
      case 'strong': return <strong key={idx} className='font-semibold text-gray-100'><InlineRenderer parts={p.v} /></strong>
      case 'em': return <em key={idx} className='italic text-gray-300'><InlineRenderer parts={p.v} /></em>
      default: return null
    }
  })}</>
}

export const Markdown: React.FC<MarkdownProps> = ({ content }) => {
  const blocks = React.useMemo(() => parseBlocks(content), [content])
  const [copiedMap, setCopiedMap] = React.useState<Record<string, boolean>>({})
  function handleCopy(key: string, text: string) {
    navigator.clipboard.writeText(text).then(() => {
      setCopiedMap(prev => ({ ...prev, [key]: true }))
  setTimeout(() => setCopiedMap(prev => { const c = { ...prev }; delete c[key]; return c }), 1800)
    })
  }
  return (
    <div className='markdown space-y-4'>
      {blocks.map(b => {
        switch (b.type) {
          case 'heading': {
            const Tag = ('h'+Math.min(6, (b as HeadingBlock).level)) as keyof JSX.IntrinsicElements
            return <Tag key={b.key} className='font-semibold text-gray-100 tracking-tight mt-2 first:mt-0'>{<InlineRenderer parts={(b as HeadingBlock).parts} />}</Tag>
          }
          case 'paragraph':
            return <p key={b.key} className='leading-relaxed text-gray-200'><InlineRenderer parts={(b as ParagraphBlock).parts} /></p>
          case 'code': {
            const cb = b as CodeBlock
            const copied = copiedMap[b.key]
            return (
              <div key={b.key} className='group relative'>
                <pre className='bg-[var(--ui-code-bg)] border border-[var(--ui-code-border)] rounded-lg p-3 overflow-auto text-[12px] leading-snug font-mono'>
                  <code>{cb.code}</code>
                </pre>
                <div className='absolute top-1 right-1 flex items-center gap-2'>
                  {cb.language && (
                    <span className='text-[10px] font-mono px-2 py-[2px] rounded bg-[#1a1a1a] border border-[#252525] text-gray-400'>{cb.language}</span>
                  )}
                  <button
                    onClick={() => handleCopy(b.key, cb.code)}
                    className={`text-[10px] px-2 py-[3px] rounded border transition-colors ${copied ? 'bg-emerald-600 border-emerald-500 text-white' : 'bg-[#1e1e1e] border-[#2a2a2a] text-gray-300 hover:bg-[#262626]'}`}
                    aria-label={copied ? 'Copied' : 'Copy code'}
                  >{copied ? 'Copied' : 'Copy'}</button>
                </div>
              </div>
            )
          }
          case 'list': {
            const lb = b as ListBlock
            if (lb.ordered) {
              return (
                <ol key={b.key} className='list-decimal list-outside ml-6 space-y-1'>
                  {lb.items.map((it, i) => <li key={i} className='pl-1'><InlineRenderer parts={it} /></li>)}
                </ol>
              )
            }
            return (
              <ul key={b.key} className='list-disc list-outside ml-6 space-y-1'>
                {lb.items.map((it, i) => <li key={i} className='pl-1'><InlineRenderer parts={it} /></li>)}
              </ul>
            )
          }
          default: return null
        }
      })}
    </div>
  )
}
