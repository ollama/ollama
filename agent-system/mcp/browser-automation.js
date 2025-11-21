/**
 * Browser Automation Module
 *
 * Enhanced Puppeteer wrapper for web automation tasks
 * Designed for agentic workflows like booking appointments, form filling, etc.
 * Optimized for M4 Mac Mini with memory-conscious design
 */

const puppeteer = require('puppeteer');

class BrowserAutomation {
    constructor(logger) {
        this.logger = logger || console;
        this.browser = null;
        this.pages = new Map();  // sessionId → { page, lastUsed, url }
        this.maxPages = 3;       // Limit concurrent pages for memory
        this.pageTimeout = 300000; // 5 minutes idle timeout
        this.cleanupInterval = null;
    }

    /**
     * Initialize the browser
     */
    async initialize() {
        if (this.browser) {
            this.logger.warn('Browser already initialized');
            return;
        }

        this.browser = await puppeteer.launch({
            headless: 'new',
            args: [
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu',
                '--no-first-run',
                '--no-zygote',
                '--single-process',
                // Memory optimization for 16GB Mac Mini
                '--js-flags=--max-old-space-size=512',
                '--disable-extensions'
            ],
            defaultViewport: {
                width: 1280,
                height: 800
            }
        });

        // Start cleanup interval
        this.cleanupInterval = setInterval(() => {
            this.cleanupIdlePages();
        }, 60000); // Check every minute

        this.logger.info('Browser automation initialized');
    }

    /**
     * Get or create a page for a session
     */
    async getPage(sessionId = 'default') {
        if (!this.browser) {
            await this.initialize();
        }

        // Check if page exists and is still valid
        if (this.pages.has(sessionId)) {
            const pageInfo = this.pages.get(sessionId);
            pageInfo.lastUsed = Date.now();

            // Check if page is still connected
            try {
                await pageInfo.page.title();
                return pageInfo.page;
            } catch (e) {
                // Page closed, remove it
                this.pages.delete(sessionId);
            }
        }

        // Enforce max pages limit
        if (this.pages.size >= this.maxPages) {
            await this.closeOldestPage();
        }

        // Create new page
        const page = await this.browser.newPage();

        // Set up page defaults
        await page.setViewport({ width: 1280, height: 800 });
        await page.setUserAgent(
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        );

        // Block unnecessary resources for speed
        await page.setRequestInterception(true);
        page.on('request', (req) => {
            const resourceType = req.resourceType();
            if (['image', 'stylesheet', 'font', 'media'].includes(resourceType)) {
                // Allow images on booking sites (often needed for verification)
                if (resourceType === 'image') {
                    req.continue();
                } else {
                    req.abort();
                }
            } else {
                req.continue();
            }
        });

        this.pages.set(sessionId, {
            page,
            lastUsed: Date.now(),
            url: null
        });

        this.logger.debug(`Created new page for session: ${sessionId}`);
        return page;
    }

    /**
     * Navigate to a URL
     */
    async navigate(url, sessionId = 'default', options = {}) {
        const page = await this.getPage(sessionId);

        try {
            await page.goto(url, {
                waitUntil: options.waitUntil || 'networkidle2',
                timeout: options.timeout || 30000
            });

            // Update page info
            const pageInfo = this.pages.get(sessionId);
            if (pageInfo) {
                pageInfo.url = url;
                pageInfo.lastUsed = Date.now();
            }

            const title = await page.title();
            const currentUrl = page.url();

            return {
                success: true,
                url: currentUrl,
                title,
                redirected: currentUrl !== url
            };
        } catch (error) {
            return {
                success: false,
                error: error.message,
                url
            };
        }
    }

    /**
     * Click an element
     */
    async click(selector, sessionId = 'default', options = {}) {
        const page = await this.getPage(sessionId);

        try {
            await page.waitForSelector(selector, {
                visible: true,
                timeout: options.timeout || 10000
            });

            await page.click(selector);

            // Wait for navigation or network idle
            if (options.waitForNavigation) {
                await page.waitForNavigation({
                    waitUntil: 'networkidle2',
                    timeout: 10000
                }).catch(() => {});
            } else {
                await page.waitForNetworkIdle({ timeout: 3000 }).catch(() => {});
            }

            return { success: true, selector };
        } catch (error) {
            return { success: false, error: error.message, selector };
        }
    }

    /**
     * Fill a form field
     */
    async fill(selector, value, sessionId = 'default', options = {}) {
        const page = await this.getPage(sessionId);

        try {
            await page.waitForSelector(selector, {
                visible: true,
                timeout: options.timeout || 10000
            });

            // Clear existing value
            if (options.clear !== false) {
                await page.click(selector, { clickCount: 3 });
                await page.keyboard.press('Backspace');
            }

            await page.type(selector, value, {
                delay: options.typeDelay || 50  // Human-like typing
            });

            return { success: true, selector, filled: true };
        } catch (error) {
            return { success: false, error: error.message, selector };
        }
    }

    /**
     * Select from dropdown
     */
    async select(selector, value, sessionId = 'default') {
        const page = await this.getPage(sessionId);

        try {
            await page.waitForSelector(selector, { timeout: 10000 });
            await page.select(selector, value);
            return { success: true, selector, selected: value };
        } catch (error) {
            return { success: false, error: error.message, selector };
        }
    }

    /**
     * Extract data from page
     */
    async extract(selector, options = {}, sessionId = 'default') {
        const page = await this.getPage(sessionId);
        const attribute = options.attribute || 'textContent';
        const multiple = options.multiple !== false;

        try {
            await page.waitForSelector(selector, { timeout: 10000 }).catch(() => {});

            const data = await page.$$eval(selector, (elements, attr) => {
                return elements.map(el => {
                    const result = {
                        text: el.textContent?.trim() || '',
                        value: el.value || '',
                        href: el.href || '',
                        id: el.id || '',
                        className: el.className || ''
                    };

                    // Get specific attribute if requested
                    if (attr && attr !== 'textContent') {
                        result[attr] = el.getAttribute(attr);
                    }

                    return result;
                });
            }, attribute);

            return {
                success: true,
                data: multiple ? data : data[0],
                count: data.length
            };
        } catch (error) {
            return { success: false, error: error.message, selector, data: [] };
        }
    }

    /**
     * Get page text content (useful for understanding page structure)
     */
    async getPageText(sessionId = 'default') {
        const page = await this.getPage(sessionId);

        try {
            const text = await page.evaluate(() => {
                // Remove script and style elements
                const clone = document.body.cloneNode(true);
                clone.querySelectorAll('script, style, noscript').forEach(el => el.remove());
                return clone.innerText;
            });

            return {
                success: true,
                text: text.substring(0, 10000), // Limit for model context
                truncated: text.length > 10000
            };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }

    /**
     * Get clickable elements (buttons, links)
     */
    async getClickableElements(sessionId = 'default') {
        const page = await this.getPage(sessionId);

        try {
            const elements = await page.evaluate(() => {
                const clickables = document.querySelectorAll(
                    'a, button, input[type="submit"], input[type="button"], [onclick], [role="button"]'
                );

                return Array.from(clickables).slice(0, 50).map((el, index) => ({
                    index,
                    tag: el.tagName.toLowerCase(),
                    text: el.textContent?.trim().substring(0, 100) || '',
                    id: el.id || null,
                    className: el.className?.toString().substring(0, 100) || '',
                    href: el.href || null,
                    type: el.type || null
                }));
            });

            return { success: true, elements };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }

    /**
     * Get form fields
     */
    async getFormFields(sessionId = 'default') {
        const page = await this.getPage(sessionId);

        try {
            const fields = await page.evaluate(() => {
                const inputs = document.querySelectorAll(
                    'input, select, textarea'
                );

                return Array.from(inputs).slice(0, 50).map((el, index) => ({
                    index,
                    tag: el.tagName.toLowerCase(),
                    type: el.type || 'text',
                    name: el.name || null,
                    id: el.id || null,
                    placeholder: el.placeholder || null,
                    label: el.labels?.[0]?.textContent?.trim() || null,
                    required: el.required || false,
                    value: el.type === 'password' ? '***' : el.value || ''
                }));
            });

            return { success: true, fields };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }

    /**
     * Wait for element
     */
    async waitFor(selector, timeout = 10000, sessionId = 'default') {
        const page = await this.getPage(sessionId);

        try {
            await page.waitForSelector(selector, { visible: true, timeout });
            return { success: true, selector, found: true };
        } catch (error) {
            return { success: false, error: error.message, selector, found: false };
        }
    }

    /**
     * Take screenshot
     */
    async screenshot(sessionId = 'default', options = {}) {
        const page = await this.getPage(sessionId);

        try {
            const buffer = await page.screenshot({
                encoding: 'base64',
                fullPage: options.fullPage || false,
                type: 'png'
            });

            return {
                success: true,
                screenshot: buffer,
                encoding: 'base64',
                mimeType: 'image/png'
            };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }

    /**
     * Execute JavaScript on page
     */
    async evaluate(script, sessionId = 'default') {
        const page = await this.getPage(sessionId);

        try {
            const result = await page.evaluate(script);
            return { success: true, result };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }

    /**
     * Scroll page
     */
    async scroll(direction = 'down', amount = 500, sessionId = 'default') {
        const page = await this.getPage(sessionId);

        try {
            await page.evaluate((dir, amt) => {
                if (dir === 'down') {
                    window.scrollBy(0, amt);
                } else if (dir === 'up') {
                    window.scrollBy(0, -amt);
                } else if (dir === 'bottom') {
                    window.scrollTo(0, document.body.scrollHeight);
                } else if (dir === 'top') {
                    window.scrollTo(0, 0);
                }
            }, direction, amount);

            return { success: true, direction, amount };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }

    /**
     * Press keyboard key
     */
    async pressKey(key, sessionId = 'default') {
        const page = await this.getPage(sessionId);

        try {
            await page.keyboard.press(key);
            return { success: true, key };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }

    /**
     * Close oldest idle page to free memory
     */
    async closeOldestPage() {
        let oldest = null;
        let oldestTime = Infinity;

        for (const [sessionId, pageInfo] of this.pages) {
            if (pageInfo.lastUsed < oldestTime) {
                oldestTime = pageInfo.lastUsed;
                oldest = sessionId;
            }
        }

        if (oldest && oldest !== 'default') {
            await this.closeSession(oldest);
        }
    }

    /**
     * Cleanup idle pages
     */
    async cleanupIdlePages() {
        const now = Date.now();

        for (const [sessionId, pageInfo] of this.pages) {
            if (now - pageInfo.lastUsed > this.pageTimeout && sessionId !== 'default') {
                this.logger.debug(`Closing idle page: ${sessionId}`);
                await this.closeSession(sessionId);
            }
        }
    }

    /**
     * Close a specific session
     */
    async closeSession(sessionId) {
        const pageInfo = this.pages.get(sessionId);
        if (pageInfo) {
            try {
                await pageInfo.page.close();
            } catch (e) {
                // Page may already be closed
            }
            this.pages.delete(sessionId);
            this.logger.debug(`Closed session: ${sessionId}`);
        }
    }

    /**
     * Get current URL
     */
    async getCurrentUrl(sessionId = 'default') {
        const page = await this.getPage(sessionId);
        return page.url();
    }

    /**
     * Shutdown browser
     */
    async shutdown() {
        if (this.cleanupInterval) {
            clearInterval(this.cleanupInterval);
        }

        for (const [sessionId] of this.pages) {
            await this.closeSession(sessionId);
        }

        if (this.browser) {
            await this.browser.close();
            this.browser = null;
        }

        this.logger.info('Browser automation shutdown complete');
    }

    /**
     * Register browser tools with ToolSystem
     */
    registerTools(toolSystem) {
        const self = this;

        toolSystem.registerTool(
            'browser_navigate',
            'Navigate browser to a URL',
            async (params) => self.navigate(params.url, params.sessionId),
            [
                { name: 'url', type: 'string', required: true, description: 'URL to navigate to' },
                { name: 'sessionId', type: 'string', required: false, description: 'Browser session ID' }
            ]
        );

        toolSystem.registerTool(
            'browser_click',
            'Click an element on the page',
            async (params) => self.click(params.selector, params.sessionId, params),
            [
                { name: 'selector', type: 'string', required: true, description: 'CSS selector to click' },
                { name: 'sessionId', type: 'string', required: false },
                { name: 'waitForNavigation', type: 'boolean', required: false }
            ]
        );

        toolSystem.registerTool(
            'browser_fill',
            'Fill a form field',
            async (params) => self.fill(params.selector, params.value, params.sessionId, params),
            [
                { name: 'selector', type: 'string', required: true, description: 'CSS selector of input' },
                { name: 'value', type: 'string', required: true, description: 'Value to enter' },
                { name: 'sessionId', type: 'string', required: false }
            ]
        );

        toolSystem.registerTool(
            'browser_select',
            'Select option from dropdown',
            async (params) => self.select(params.selector, params.value, params.sessionId),
            [
                { name: 'selector', type: 'string', required: true },
                { name: 'value', type: 'string', required: true },
                { name: 'sessionId', type: 'string', required: false }
            ]
        );

        toolSystem.registerTool(
            'browser_extract',
            'Extract data from page elements',
            async (params) => self.extract(params.selector, params, params.sessionId),
            [
                { name: 'selector', type: 'string', required: true, description: 'CSS selector' },
                { name: 'attribute', type: 'string', required: false, description: 'Attribute to extract' },
                { name: 'sessionId', type: 'string', required: false }
            ]
        );

        toolSystem.registerTool(
            'browser_get_text',
            'Get all text content from the page',
            async (params) => self.getPageText(params.sessionId),
            [
                { name: 'sessionId', type: 'string', required: false }
            ]
        );

        toolSystem.registerTool(
            'browser_get_clickables',
            'Get all clickable elements on the page',
            async (params) => self.getClickableElements(params.sessionId),
            [
                { name: 'sessionId', type: 'string', required: false }
            ]
        );

        toolSystem.registerTool(
            'browser_get_forms',
            'Get all form fields on the page',
            async (params) => self.getFormFields(params.sessionId),
            [
                { name: 'sessionId', type: 'string', required: false }
            ]
        );

        toolSystem.registerTool(
            'browser_screenshot',
            'Take a screenshot of the page',
            async (params) => self.screenshot(params.sessionId, params),
            [
                { name: 'sessionId', type: 'string', required: false },
                { name: 'fullPage', type: 'boolean', required: false }
            ]
        );

        toolSystem.registerTool(
            'browser_scroll',
            'Scroll the page',
            async (params) => self.scroll(params.direction, params.amount, params.sessionId),
            [
                { name: 'direction', type: 'string', required: true, description: 'up, down, top, bottom' },
                { name: 'amount', type: 'number', required: false, description: 'Pixels to scroll' },
                { name: 'sessionId', type: 'string', required: false }
            ]
        );

        this.logger.info('Browser automation tools registered with ToolSystem');
    }
}

module.exports = BrowserAutomation;
