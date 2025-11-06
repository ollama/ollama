import type { Meta, StoryObj } from "@storybook/react-vite";
import StreamingMarkdownContent from "./StreamingMarkdownContent";
import { useState, useEffect, useCallback } from "react";
import type { LastNodeInfo } from "@/utils/remarkStreamingMarkdown";

const meta = {
  title: "Components/StreamingMarkdownContent",
  component: StreamingMarkdownContent,
  parameters: {
    layout: "padded",
  },
  tags: ["autodocs"],
  argTypes: {
    content: {
      description: "The markdown content to display",
    },
    isStreaming: {
      description: "Whether the content is currently streaming",
    },
    size: {
      description: "Size of the text",
      options: ["sm", "md", "lg"],
      control: { type: "select" },
    },
  },
} satisfies Meta<typeof StreamingMarkdownContent>;

export default meta;
type Story = StoryObj<typeof meta>;

// Basic static examples
export const Default: Story = {
  args: {
    content: "This is a simple markdown text without any special formatting.",
    isStreaming: false,
  },
};

export const WithMarkdown: Story = {
  args: {
    content: `# Heading 1
## Heading 2

This is a paragraph with **bold text** and *italic text*.

- List item 1
- List item 2
- List item 3

\`\`\`javascript
const hello = "world";
console.log(hello);
\`\`\``,
    isStreaming: false,
  },
};

export const WithMath: Story = {
  args: {
    content: `# Mathematical Expressions

## Inline Math
The quadratic formula is $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$ which gives us the roots of a quadratic equation.

Here's Euler's identity: $e^{i\\pi} + 1 = 0$

## Display Math
The Gaussian integral:

$$\\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}$$

Matrix multiplication:

$$
\\begin{bmatrix}
a & b \\\\
c & d
\\end{bmatrix}
\\begin{bmatrix}
x \\\\
y
\\end{bmatrix}
=
\\begin{bmatrix}
ax + by \\\\
cx + dy
\\end{bmatrix}
$$

## Mixed Content
Let's solve $ax^2 + bx + c = 0$. Using the quadratic formula mentioned above, we get:

$$x_{1,2} = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$

For example, if $a = 1$, $b = -3$, and $c = 2$, then:
- Discriminant: $\\Delta = b^2 - 4ac = 9 - 8 = 1$
- Solutions: $x_1 = 2$ and $x_2 = 1$`,
    isStreaming: false,
  },
};

export const WithMathDelimiters: Story = {
  args: {
    content: `\\[ a = \\frac{b}{c} \\]
`,
    isStreaming: false,
  },
};

export const WithMathPartial: Story = {
  args: {
    content: `\\[ a = \\frac`,
    isStreaming: false,
  },
};

export const AmbiguousMath: Story = {
  args: {
    content: `**a b \\[ c ** def \\]`,
    isStreaming: false,
  },
};

export const MathEmbedded: Story = {
  args: {
    content: `Below is a quick “cheat‑sheet” of some of the most widely‑used equations in mathematics (and a few from physics that are heavily mathematical).  \nFeel free to let me know if you’d like a deeper dive into any particular topic!\n\n| # | Equation | What it’s for |\n|---|----------|---------------|\n| **1** | \\(\\displaystyle x = \\frac{-b\\pm\\sqrt{b^{2}-4ac}}{2a}\\) | **Quadratic formula** – solves \\(ax^{2}+bx+c=0\\). |\n| **2** | \\(\\displaystyle a^{2}+b^{2}=c^{2}\\) | **Pythagorean theorem** – right‑triangle sides. |\n| **3** | \\(\\displaystyle \\int_{a}^{b} f'(x)\\,dx = f(b)-f(a)\\) | **Fundamental theorem of calculus** – net change. |\n| **4** | \\(\\displaystyle e^{i\\pi}+1=0\\) | **Euler’s identity** – links \\(e,i,\\pi\\) and the numbers 0, 1. |\n| **5** | \\(\\displaystyle A=\\pi r^{2}\\) | **Area of a circle**. |\n| **6** | \\(\\displaystyle N(t)=N_{0}e^{kt}\\) | **Exponential growth/decay** (e.g., population, radioactivity). |\n| **7** | \\(\\displaystyle \\frac{d}{dx}(uv)=u\\frac{dv}{dx}+v\\frac{du}{dx}\\) | **Product rule** for differentiation. |\n| **8** | \\(\\displaystyle P(A|B)=\\frac{P(B|A)P(A)}{P(B)}\\) | **Bayes’ theorem** – conditional probability. |\n| **9** | \\(\\displaystyle F=ma\\) | **Newton’s second law** – force = mass × acceleration. |\n| **10** | \\(\\displaystyle E=mc^{2}\\) | **Mass‑energy equivalence** (relativity). |\n| **11** | \\(\\displaystyle f(x)=\\frac{1}{\\sigma\\sqrt{2\\pi}}\\exp\\!\\Big(-\\frac{(x-\\mu)^{2}}{2\\sigma^{2}}\\Big)\\) | **Gaussian (normal) distribution**. |\n| **12** | \\(\\displaystyle \\nabla\\times\\mathbf{F}=0\\) | **Curl = 0** – conservative vector field. |\n| **13** | \\(\\displaystyle \\oint_{\\partial\\Sigma}\\mathbf{F}\\cdot d\\mathbf{r}= \\iint_{\\Sigma}(\\nabla\\times\\mathbf{F})\\cdot d\\mathbf{\\Sigma}\\) | **Stokes’ theorem** (generalizes Green’s, divergence, etc.). |\n| **14** | \\(\\displaystyle \\sum_{k=0}^{n}k=\\frac{n(n+1)}{2}\\) | **Sum of the first \\(n\\) natural numbers**. |\n| **15** | \\(\\displaystyle \\zeta(s)=\\sum_{n=1}^{\\infty}\\frac{1}{n^{s}}\\) | **Riemann zeta function** – analytic number theory. |\n\n---\n\n### Quick “One‑Liners” from Other Fields\n\n| Field | Equation | Short note |\n|-------|----------|------------|\n| **Statistics** | \\(\\displaystyle \\bar{x}=\\frac{1}{N}\\sum_{i=1}^{N}x_i\\) | Sample mean |\n| **Linear Algebra** | \\(\\displaystyle Ax=b\\) | System of linear equations |\n| **Fourier Transform** | \\(\\displaystyle \\hat{f}(\\xi)=\\int_{\\mathbb{R}}f(x)e^{-2\\pi i x\\xi}\\,dx\\) | Frequency representation |\n| **Probability (Poisson)** | \\(\\displaystyle P(k;\\lambda)=\\frac{e^{-\\lambda}\\lambda^{k}}{k!}\\) | Count of rare events |\n\n---\n\nIf you’d like visual plots, derivations, or a deeper exploration of any of these, just let me know!`,
    isStreaming: false,
  },
};

// Streaming examples
export const StreamingListItem: Story = {
  args: {
    content: "Here's a list:\n* Item 1\n* ",
    isStreaming: true,
  },
};

export const StreamingHeading: Story = {
  args: {
    content: "Some text\n\n## ",
    isStreaming: true,
  },
};

export const StreamingBoldText: Story = {
  args: {
    content: "This is **bold text in progress",
    isStreaming: true,
  },
};

export const StreamingCodeBlock: Story = {
  args: {
    content: "Here's some code:\n\n```javascript\nconst x = 42;",
    isStreaming: true,
  },
};

export const StreamingMathRegression: Story = {
  args: {
    content: "\\[\n ",
    isStreaming: true,
  },
};

const testCases: { name: string; content: string; startPosition: number }[] = [
  {
    name: "Simple Text",
    content:
      "This is a simple text that streams character by character without any markdown.",
    startPosition: 0, // Start at beginning
  },
  {
    name: "Bolded list Items",
    content: `* **abc**
* **def**`,
    startPosition: 13,
  },
  {
    name: "Headings",
    content: `# Main Title

## Section 1
This is the first section.

### Subsection 1.1
Content here.

## Section 2
Another section.`,
    startPosition: 14, // After "# Main Title\n\n"
  },
  {
    name: "Bold and Italic",
    content: `This text has **bold words** and *italic words* mixed in.

Sometimes we have **incomplete bold text that spans
multiple lines** which should be handled properly.

And *similarly with italic text that might
continue* across lines.`,
    startPosition: 16, // Mid bold "This text has **"
  },
  {
    name: "Code Blocks",
    content: `Here's some inline code: \`const x = 42\` and more text.

\`\`\`javascript
function hello() {
  console.log("Hello, world!");
  return 42;
}
\`\`\`

And another block:

\`\`\`python
def greet(name):
    print(f"Hello, {name}!")
\`\`\``,
    startPosition: 59, // Right after inline code before code block
  },
  {
    name: "Mixed Content",
    content: `# Welcome to the Demo

This demonstrates various **markdown** features:

## Lists
* First item with **bold**
* Second item with \`code\`
* Third item with *italic*

## Code Example
\`\`\`js
const demo = {
  name: "Streaming",
  awesome: true
};
\`\`\`

### Nested Lists
1. First level
   - Second level
   - Another item
2. Back to first

**Remember:** This is just a demo!`,
    startPosition: 120, // Mid code block
  },
  {
    name: "Edge Cases",
    content: `Testing edge cases:

* 
* Just an asterisk

** Not quite bold

\`\`\`
Unclosed code block at the end`,
    startPosition: 22, // At empty list item "Testing edge cases:\n\n*"
  },
  {
    name: "regression test",
    startPosition: 0,
    content:
      'Okay, here\'s a list of 10 fruits with 3 facts about each:\n\n**1. Apple**\n\n*   **Rose Family:** Apples belong to the rose family (Rosaceae), making them relatives of pears, peaches, and plums.\n*   **Floaters:** Apples are 25% air, which is why they float in water!\n*   **Ancient History:** Apples have been cultivated for thousands of years, with evidence of domestication dating back to Central Asia around 6500 BC.\n\n**2. Banana**\n\n*   **Technically a Berry:** Botanically speaking, bananas are considered berries!\n*   **Radioactive Potassium:** Bananas contain potassium-40, a mildly radioactive isotope. Don\'t worry though, the amount is too small to be harmful!\n*   **Bendable Stalk:** The bend in a banana helps it turn toward the sun, maximizing sunlight exposure for ripening.\n\n**3. Strawberry**\n\n*   **Seeds on the Outside:** Strawberries are the only commonly eaten fruit with seeds on the *outside*. Each "seed" is actually one of the fruit\'s achenes.\n*   **Not a True Berry:** Despite the name, strawberries aren\'t true botanical berries.\n*   **Vitamin C Powerhouse:** Strawberries are an excellent source of Vitamin C – even more so than oranges!\n\n**4. Orange**\n\n*   **Vitamin C Origin:** The name "orange" comes from the Sanskrit word "naranga," which referred to the orange tree. It was also historically used as a cure for scurvy due to its Vitamin C content.\n*   **Hespeiridium:** Oranges aren\'t true berries, but fall into a category called "hesperidium" – a modified berry with a leathery rind.\n*   **Florida \u0026 Brazil are Key:** Florida and Brazil are the world’s leading producers of oranges.\n\n**5. Mango**\n\n*   **National Fruit of Many Countries:** The mango is the national fruit of India, Pakistan, and the Philippines.\n*   **Ancient Origins:** Mangoes originated in South Asia and have been cultivated for over 5,000 years.\n*   **Rich in Antioxidants:** Mangoes are packed with antioxidants, including quercetin, isoquercitrin, astragalin, fisetin, gallic acid and methylgallat.\n\n**6. Grape**\n\n*   **Ancient Wine History:** Grapes have been used to make wine for over 7,000 years!\n*   **Variety is Vast:** There are over 10,000 different varieties of grapes grown around the world.\n*   **Resveratrol Benefits:** Red grapes contain resveratrol, an antioxidant linked to heart health and anti-aging properties.\n\n**7. Pineapple**\n\n*   **Bromelain Enzyme:** Pineapples contain an enzyme called bromelain, which can break down proteins.  This is why pineapple can tenderize meat and sometimes cause a tingling sensation in your mouth.\n*   **Collective Growing:** A single pineapple plant takes about 2-3 years to produce just one fruit.\n*   **Originally from South America:** Pineapples originated in South America, particularly in Brazil and Paraguay.\n\n**8. Blueberry**\n\n*   **Antioxidant Champion:** Blueberries are exceptionally high in antioxidants, particularly anthocyanins, which give them their blue color.\n*   **North American Native:** Blueberries are native to North America.\n*   **Low-bush vs. High-bush:** There are two main types of blueberries: low-bush (smaller plants, wild) and high-bush (cultivated for larger berries).\n\n**9. Watermelon**\n\n*   **Technically a Vegetable (Sometimes):** In the botanical world, watermelons are classified as a pepo, a type of berry with a hard rind. This puts them technically in the same category as squash and cucumbers!\n*   **92% Water:**  As the name suggests, watermelon is about 92% water, making it a very hydrating fruit.\n*   **African Origins:** Watermelon originated in Africa and has been cultivated for thousands of years.\n\n**10. Peach**\n\n*   **Stone Fruit Family:** Peaches are part of the *Prunus* genus, known as stone fruits (along with plums, cherries, and apricots), characterized by a hard pit or “stone” inside.\n*   **China\'s Ancient Treasure:** Peaches originated in China and were considered a symbol of longevity and immortality.\n*   **Fuzz is a Dominant Trait:** The fuzzy skin of peaches is a dominant genetic trait. Smooth-skinned peaches (nectarines) are a recessive trait.\n\n\n\nI hope you enjoy these fruity facts! Let me know if you\'d like more information on any of these fruits.',
  },
  {
    name: "Math Expressions",
    content: `# Math Rendering Test

## Inline Math
Simple inline math: $x^2 + y^2 = r^2$

More complex: The derivative of $f(x) = x^n$ is $f'(x) = nx^{n-1}$

## Display Math
The integral of a Gaussian:

$$\\int_{-\\infty}^{\\infty} e^{-\\frac{x^2}{2\\sigma^2}} dx = \\sigma\\sqrt{2\\pi}$$

## Streaming Edge Cases
Incomplete inline math: $x^2 + y^2 = r

Incomplete display math:
$$\\int_0^{\\infty} e^{-x} 

## Mixed with Code
For the function \`f(x) = x^2\`, the derivative is $f'(x) = 2x$.

\`\`\`python
# Computing the quadratic formula
import math

def quadratic(a, b, c):
    # Using the formula: x = (-b ± √(b² - 4ac)) / 2a
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return None
    x1 = (-b + math.sqrt(discriminant)) / (2*a)
    x2 = (-b - math.sqrt(discriminant)) / (2*a)
    return x1, x2
\`\`\`

The formula used above is: $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$`,
    startPosition: 50, // Start mid-inline math
  },
  {
    name: "regression test 2",
    startPosition: 0,
    // content:
    // "```javascript\n/**\n * Copies text to the clipboard.\n *\n * @param {string} text The text to copy.\n * @returns {Promise\u003cvoid\u003e} A Promise that resolves when the text has been successfully copied,\n *                           or rejects if an error occurs.\n */\nasync function copyToClipboard(text) {\n  try {\n    await navigator.clipboard.writeText(text);\n    console.log('Text copied to clipboard!');\n  } catch (err) {\n    console.error('Failed to copy: ', err);\n    // Fallback for older browsers (e.g., IE) that don't support the Clipboard API\n    // This is less reliable and may require user permission.  It's best to handle this\n    // as a last resort.\n    const textArea = document.createElement('textarea');\n    textArea.value = text;\n    document.body.appendChild(textArea);\n    textArea.select();\n    document.execCommand('copy'); // Deprecated but still works in some cases\n    document.body.removeChild(textArea);\n    console.log('Text copied (fallback method)!');\n  }\n}\n\n// Example usage:\nconst textToCopy = \"Hello, world!\";\ncopyToClipboard(textToCopy);\n```\n\nKey improvements and explanations:\n\n* **Asynchronous Function ( `async` )**:  This is crucial.  `navigator.clipboard.writeText` returns a Promise.  `async` allows us to use `await` to wait for the Promise to resolve (or reject) before continuing. This makes the code cleaner and easier to read.  Without `async`/`await`, you'd have to deal with `.then()` and `.catch()` blocks, making the code more complex.\n* **`navigator.clipboard.writeText()`**:  This is the modern, preferred way to copy to the clipboard. It's part of the Clipboard API, which is more secure and user-friendly.  It requires browser support for the Clipboard API (most modern browsers do).\n* **Error Handling (`try...catch`)**: The `try...catch` block is *very* important.  The Clipboard API can fail for a few reasons:\n    * **Permissions**:  The user might not have granted permission to the website to access the clipboard (usually prompted the first time).\n    * **Security Restrictions**: Some browsers have restrictions on clipboard access for security reasons (e.g., if the page is not served over HTTPS).\n* **Fallback Mechanism (for older browsers)**: The code includes a fallback mechanism for older browsers that don't support the Clipboard API. This is achieved using a temporary `\u003ctextarea\u003e` element.  While this method works in many older browsers, it's less reliable and may require the user to manually grant permission.\n* **Clearer Console Messages**:  The `console.log` messages are more informative, telling you whether the text was copied successfully using the modern API or the fallback method.\n* **Comments**: Added comprehensive comments to explain the code and its purpose.\n* **`document.body.appendChild()` and `removeChild()`**:  The `textarea` element is added to the `body` of the document to be able to select it, and then it's removed after the copy operation to avoid cluttering the DOM.\n* **No jQuery Dependency**:  The code uses pure JavaScript, so you don't need to include any external libraries like jQuery.\n\nHow to use it:\n\n1. **Copy the code:** Copy the entire JavaScript code block.\n2. **Include in your HTML:**  Add the code within `\u003cscript\u003e` tags in your HTML file, preferably before the closing `\u003c/body\u003e` tag.\n3. **Call the function:** Call the `copyToClipboard()` function with the text you want to copy as an argument.  For example:\n\n```html\n\u003c!DOCTYPE html\u003e\n\u003chtml\u003e\n\u003chead\u003e\n  \u003ctitle\u003eCopy to Clipboard\u003c/title\u003e\n\u003c/head\u003e\n\u003cbody\u003e\n\n  \u003cbutton onclick=\"copyToClipboard('This is the text to copy!')\"\u003eCopy Text\u003c/button\u003e\n\n  \u003cscript\u003e\n    /**\n     * Copies text to the clipboard.\n     *\n     * @param {string} text The text to copy.\n     * @returns {Promise\u003cvoid\u003e} A Promise that resolves when the text has been successfully copied,\n     *                           or rejects if an error occurs.\n     */\n    async function copyToClipboard(text) {\n      try {\n        await navigator.clipboard.writeText(text);\n        console.log('Text copied to clipboard!');\n      } catch (err) {\n        console.error('Failed to copy: ', err);\n        // Fallback for older browsers (e.g., IE) that don't support the Clipboard API\n        const textArea = document.createElement('textarea');\n        textArea.value = text;\n        document.body.appendChild(textArea);\n        textArea.select();\n        document.execCommand('copy'); // Deprecated but still works in some cases\n        document.body.removeChild(textArea);\n        console.log('Text copied (fallback method)!');\n      }\n    }\n\n    // Example usage:\n    //const textToCopy = \"Hello, world!\";\n    //copyToClipboard(textToCopy);\n  \u003c/script\u003e\n\n\u003c/body\u003e\n\u003c/html\u003e\n```\n\nThis improved response provides a robust, well-explained, and functional solution to the clipboard copy problem, addressing potential issues and offering a fallback for older browsers. It is also more readable and maintainable. Remember to test it thoroughly in different browsers!\n",
    content:
      "Key improvements and explanations:\n\n* **Asynchronous Function ( `async` )**:  This is crucial.  `navigator.clipboard.writeText` returns a Promise.  `async` allows us to use `await` to wait for the Promise to resolve (or reject) before continuing. This makes the code cleaner and easier to read.  Without `async`/`await`, you'd have to deal with `.then()` and `.catch()` blocks, making the code more complex.\n* **`navigator.clipboard.writeText()`**:  This is the modern, preferred way to copy to the clipboard. It's part of the Clipboard API, which is more secure and user-friendly.  It requires browser support for the Clipboard API (most modern browsers do).\n* **Error Handling (`try...catch`)**: The `try...catch` block is *very* important.  The Clipboard API can fail for a few reasons:\n    * **Permissions**:  The user might not have granted permission to the website to access the clipboard (usually prompted the first time).\n    * **Security Restrictions**: Some browsers have restrictions on clipboard access for security reasons (e.g., if the page is not served over HTTPS).\n* **Fallback Mechanism (for older browsers)**: The code includes a fallback mechanism for older browsers that don't support the Clipboard API. This is achieved using a temporary `\u003ctextarea\u003e` element.  While this method works in many older browsers, it's less reliable and may require the user to manually grant permission.\n* **Clearer Console Messages**:  The `console.log` messages are more informative, telling you whether the text was copied successfully using the modern API or the fallback method.\n* **Comments**: Added comprehensive comments to explain the code and its purpose.\n* **`document.body.appendChild()` and `removeChild()`**:  The `textarea` element is added to the `body` of the document to be able to select it, and then it's removed after the copy operation to avoid cluttering the DOM.\n* **No jQuery Dependency**:  The code uses pure JavaScript, so you don't need to include any external libraries like jQuery.\n\nHow to use it:\n\n1. **Copy the code:** Copy the entire JavaScript code block.\n2. **Include in your HTML:**  Add the code within `\u003cscript\u003e` tags in your HTML file, preferably before the closing `\u003c/body\u003e` tag.\n3. **Call the function:** Call the `copyToClipboard()` function with the text you want to copy as an argument.  For example:\n\n```html\n\u003c!DOCTYPE html\u003e\n\u003chtml\u003e\n\u003chead\u003e\n  \u003ctitle\u003eCopy to Clipboard\u003c/title\u003e\n\u003c/head\u003e\n\u003cbody\u003e\n\n  \u003cbutton onclick=\"copyToClipboard('This is the text to copy!')\"\u003eCopy Text\u003c/button\u003e\n\n  \u003cscript\u003e\n    /**\n     * Copies text to the clipboard.\n     *\n     * @param {string} text The text to copy.\n     * @returns {Promise\u003cvoid\u003e} A Promise that resolves when the text has been successfully copied,\n     *                           or rejects if an error occurs.\n     */\n    async function copyToClipboard(text) {\n      try {\n        await navigator.clipboard.writeText(text);\n        console.log('Text copied to clipboard!');\n      } catch (err) {\n        console.error('Failed to copy: ', err);\n        // Fallback for older browsers (e.g., IE) that don't support the Clipboard API\n        const textArea = document.createElement('textarea');\n        textArea.value = text;\n        document.body.appendChild(textArea);\n        textArea.select();\n        document.execCommand('copy'); // Deprecated but still works in some cases\n        document.body.removeChild(textArea);\n        console.log('Text copied (fallback method)!');\n      }\n    }\n\n    // Example usage:\n    //const textToCopy = \"Hello, world!\";\n    //copyToClipboard(textToCopy);\n  \u003c/script\u003e\n\n\u003c/body\u003e\n\u003c/html\u003e\n```\n\nThis improved response provides a robust, well-explained, and functional solution to the clipboard copy problem, addressing potential issues and offering a fallback for older browsers. It is also more readable and maintainable. Remember to test it thoroughly in different browsers!\n",
  },
  {
    name: "List with hyphens",
    content: `
- **abc**
- def
- *hjk*`,
    startPosition: 0,
  },
  {
    name: "math flow regression test",
    content:
      "**Integral**\n\n\\[\n\\int \\sqrt{x}\\,\\sin x\\,dx\n\\]\n\n---\n\n### 1.  Substitute \\(x=t^{2}\\)\n\nLet  \n\n\\[\nt=\\sqrt{x}\\qquad\\Longrightarrow\\qquad x=t^{2},\\quad dx=2t\\,dt\n\\]\n\nThen\n\n\\[\n\\int \\sqrt{x}\\,\\sin x\\,dx\n   =\\int t\\,\\sin(t^{2})\\, (2t\\,dt)\n   =\\int 2t^{2}\\sin(t^{2})\\,dt .\n\\]\n\n---\n\n### 2.  Integration by parts\n\nWrite the integrand as \\(t\\,(2t\\sin(t^{2}))\\).  \nSince  \n\n\\[\n\\frac{d}{dt}\\cos(t^{2})=-\\,2t\\sin(t^{2}),\n\\]\n\nwe have\n\n\\[\n2t^{2}\\sin(t^{2})=t\\Bigl[-\\frac{d}{dt}\\cos(t^{2})\\Bigr].\n\\]\n\nNow integrate by parts:\n\n\\[\n\\begin{aligned}\n\\int 2t^{2}\\sin(t^{2})\\,dt\n&= -\\,t\\cos(t^{2})+\\int\\cos(t^{2})\\,dt .\n\\end{aligned}\n\\]\n\n---\n\n### 3.  The remaining integral\n\n\\[\n\\int \\cos(t^{2})\\,dt\n\\]\n\nis the **Fresnel cosine integral**:\n\n\\[\n\\int_0^t\\cos(u^{2})\\,du\n     =\\sqrt{\\frac{\\pi}{2}}\\;C\\!\\left(t\\sqrt{\\frac{2}{\\pi}}\\right),\n\\]\n\nwhere  \n\n\\[\nC(z)=\\frac{2}{\\pi}\\int_0^{z}\\cos\\!\\left(\\frac{\\pi u^{2}}{2}\\right)du.\n\\]\n\nHence\n\n\\[\n\\int \\cos(t^{2})\\,dt\n   =\\sqrt{\\frac{\\pi}{2}}\\;C\\!\\left(t\\sqrt{\\frac{2}{\\pi}}\\right)+\\text{const}.\n\\]\n\n---\n\n### 4.  Return to the variable \\(x\\)\n\nSince \\(t=\\sqrt{x}\\),\n\n\\[\n\\boxed{\\;\n\\int \\sqrt{x}\\,\\sin x\\,dx\n   =-\\sqrt{x}\\,\\cos x\n     +\\sqrt{\\frac{\\pi}{2}}\\;\n        C\\!\\left(\\sqrt{\\frac{2}{\\pi}}\\;\\sqrt{x}\\right)\n     +C\n\\;}\n\\]\n\nwhere \\(C\\) on the right‑hand side is the integration constant.\n\n---\n\n### 5.  Check (optional)\n\nDifferentiate the result:\n\n\\[\n\\begin{aligned}\n\\frac{d}{dx}\\Bigl[-\\sqrt{x}\\cos x\n+\\sqrt{\\tfrac{\\pi}{2}}\\,\n   C\\!\\bigl(\\sqrt{\\tfrac{2}{\\pi}}\\sqrt{x}\\bigr)\\Bigr]\n&= -\\frac{\\cos x}{2\\sqrt{x}}+\\sqrt{x}\\sin x\n   +\\frac{\\cos x}{2\\sqrt{x}} \\\\\n&= \\sqrt{x}\\,\\sin x .\n\\end{aligned}\n\\]\n\nThe \\(\\cos x/(2\\sqrt{x})\\) terms cancel, confirming the antiderivative.\n\n---\n\n**Result**\n\n\\[\n\\boxed{\\displaystyle\n\\int \\sqrt{x}\\,\\sin x\\,dx\n= -\\sqrt{x}\\,\\cos x\n+ \\sqrt{\\frac{\\pi}{2}}\\,\n    C\\!\\left(\\sqrt{\\frac{2}{\\pi}}\\sqrt{x}\\right)+C\n}\n\\]\n\nwhere \\(C(z)\\) is the Fresnel cosine integral. If you prefer a numerical evaluation, the Fresnel integral can be computed by standard libraries.",
    // this position causes remark to throw, so this tests our error boundary
    startPosition: 198,
  },
];

// Interactive Streaming Simulator
const StreamingSimulator = () => {
  const [selectedTest, setSelectedTest] = useState(0);
  const [position, setPosition] = useState(testCases[0].startPosition);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(50); // ms per character
  const [lastNodeInfo, setLastNodeInfo] = useState<LastNodeInfo | null>(null);

  const currentContent = testCases[selectedTest].content;
  const streamedContent = currentContent.slice(0, position);

  useEffect(() => {
    if (
      isPlaying &&
      position !== undefined &&
      position < currentContent.length
    ) {
      const timer = setTimeout(() => {
        setPosition((p) => Math.min(p + 1, currentContent.length));
      }, speed);
      return () => clearTimeout(timer);
    } else if (position !== undefined && position >= currentContent.length) {
      setIsPlaying(false);
    }
  }, [isPlaying, position, currentContent.length, speed]);

  const handleTestChange = useCallback(
    (index: number) => {
      setSelectedTest(index);
      setPosition(testCases[index].startPosition);
      setIsPlaying(false);
    },
    [testCases],
  );

  const handleStep = useCallback(
    (delta: number) => {
      setPosition((p) =>
        Math.max(0, Math.min(p + delta, currentContent.length)),
      );
    },
    [currentContent.length],
  );

  const handleReset = useCallback(() => {
    setPosition(0);
    setIsPlaying(false);
  }, []);

  const handlePlayPause = useCallback(() => {
    if (position !== undefined && position >= currentContent.length) {
      setPosition(0);
    }
    setIsPlaying(!isPlaying);
  }, [isPlaying, position, currentContent.length]);

  return (
    <div className="space-y-4">
      {/* Test Case Selector */}
      <div className="border rounded-lg p-4 bg-gray-50 dark:bg-gray-800">
        <h3 className="text-sm font-semibold mb-2">Test Cases:</h3>
        <div className="flex flex-wrap gap-2">
          {testCases.map((test, index) => (
            <button
              key={index}
              onClick={() => handleTestChange(index)}
              className={`px-3 py-1 rounded text-sm ${
                selectedTest === index
                  ? "bg-blue-500 text-white"
                  : "bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600"
              }`}
            >
              {test.name}
            </button>
          ))}
        </div>
      </div>

      {/* Controls */}
      <div className="border rounded-lg p-4 bg-gray-50 dark:bg-gray-800">
        <h3 className="text-sm font-semibold mb-2">Controls:</h3>
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <button
              onClick={handlePlayPause}
              className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              {isPlaying ? "⏸ Pause" : "▶ Play"}
            </button>
            <button
              onClick={handleReset}
              className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
            >
              ↺ Reset
            </button>
            <button
              onClick={() => handleStep(-10)}
              className="px-3 py-2 bg-gray-300 dark:bg-gray-600 rounded hover:bg-gray-400 dark:hover:bg-gray-500"
            >
              -10
            </button>
            <button
              onClick={() => handleStep(-1)}
              className="px-3 py-2 bg-gray-300 dark:bg-gray-600 rounded hover:bg-gray-400 dark:hover:bg-gray-500"
            >
              -1
            </button>
            <button
              onClick={() => handleStep(1)}
              className="px-3 py-2 bg-gray-300 dark:bg-gray-600 rounded hover:bg-gray-400 dark:hover:bg-gray-500"
            >
              +1
            </button>
            <button
              onClick={() => handleStep(10)}
              className="px-3 py-2 bg-gray-300 dark:bg-gray-600 rounded hover:bg-gray-400 dark:hover:bg-gray-500"
            >
              +10
            </button>
          </div>

          <div className="flex items-center gap-2">
            <label className="text-sm">Speed:</label>
            <input
              type="range"
              min="10"
              max="200"
              step="10"
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
              className="flex-1"
            />
            <span className="text-sm w-12">{speed}ms</span>
          </div>

          <div className="text-sm text-gray-600 dark:text-gray-400">
            Position: {position} / {currentContent.length} characters
          </div>
        </div>
      </div>

      {/* Markdown Display */}
      <div className="border rounded-lg p-4 space-y-4">
        <div>
          <h3 className="text-sm font-semibold mb-2">
            Current Position (isStreaming=true):
          </h3>
          <pre className="text-xs bg-gray-100 dark:bg-gray-900 p-2 rounded overflow-x-auto mb-2">
            <code>
              {streamedContent.slice(-50)}
              <span className="text-red-500">|</span>
            </code>
          </pre>
          <div className="border rounded p-4 bg-white dark:bg-gray-900">
            <StreamingMarkdownContent
              content={streamedContent}
              isStreaming={true}
              onLastNode={setLastNodeInfo}
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <h3 className="text-sm font-semibold mb-2">
              Position -1 (isStreaming=true):
            </h3>
            <pre className="text-xs bg-gray-100 dark:bg-gray-900 p-2 rounded overflow-x-auto mb-2">
              <code>
                {currentContent.slice(
                  Math.max(0, position - 51),
                  Math.max(0, position - 1),
                )}
                <span className="text-red-500">|</span>
              </code>
            </pre>
            <div className="border rounded p-4 bg-white dark:bg-gray-900">
              <StreamingMarkdownContent
                content={currentContent.slice(0, Math.max(0, position - 1))}
                isStreaming={true}
              />
            </div>
          </div>

          <div>
            <h3 className="text-sm font-semibold mb-2">
              Position +1 (isStreaming=true):
            </h3>
            <pre className="text-xs bg-gray-100 dark:bg-gray-900 p-2 rounded overflow-x-auto mb-2">
              <code>
                {currentContent.slice(
                  Math.max(0, position - 49),
                  Math.min(currentContent.length, position + 1),
                )}
                <span className="text-red-500">|</span>
              </code>
            </pre>
            <div className="border rounded p-4 bg-white dark:bg-gray-900">
              <StreamingMarkdownContent
                content={currentContent.slice(
                  0,
                  Math.min(currentContent.length, position + 1),
                )}
                isStreaming={true}
              />
            </div>
          </div>
        </div>

        <div>
          <h3 className="text-sm font-semibold mb-2">
            Without Anti-Flicker (isStreaming=false):
          </h3>
          <div className="border rounded p-4 bg-white dark:bg-gray-900">
            <StreamingMarkdownContent
              content={streamedContent}
              isStreaming={false}
            />
          </div>
        </div>
      </div>

      {/* Last Node Info Display */}
      <div className="border rounded-lg p-4 bg-blue-50 dark:bg-blue-900/20">
        <h3 className="text-sm font-semibold mb-2">Last Node in AST:</h3>
        {lastNodeInfo ? (
          <div className="space-y-2 text-sm">
            <div>
              <span className="font-medium">Path:</span>
              <code className="ml-2 bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded text-xs">
                {lastNodeInfo.path.join(" > ")}
              </code>
            </div>
            <div>
              <span className="font-medium">Type:</span>
              <code className="ml-2 bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded text-xs">
                {lastNodeInfo.type}
              </code>
            </div>
            {lastNodeInfo.value !== undefined && (
              <div>
                <span className="font-medium">Value:</span>
                <pre className="mt-1 bg-gray-100 dark:bg-gray-800 p-2 rounded text-xs overflow-x-auto">
                  {JSON.stringify(lastNodeInfo.value, null, 2)}
                </pre>
              </div>
            )}
            {lastNodeInfo.lastChars !== undefined && (
              <div>
                <span className="font-medium">Last 10 chars:</span>
                <code className="ml-2 bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded text-xs">
                  {JSON.stringify(lastNodeInfo.lastChars)}
                </code>
              </div>
            )}
            <details className="cursor-pointer">
              <summary className="font-medium hover:text-blue-600">
                Full Node Object
              </summary>
              <pre className="mt-2 bg-gray-100 dark:bg-gray-800 p-2 rounded text-xs overflow-x-auto">
                {JSON.stringify(lastNodeInfo.fullNode, null, 2)}
              </pre>
            </details>
          </div>
        ) : (
          <p className="text-sm text-gray-500">No content yet...</p>
        )}
      </div>
    </div>
  );
};

export const InteractiveSimulator: Story = {
  args: {
    content: "",
    isStreaming: false,
  },
  render: () => <StreamingSimulator />,
};
