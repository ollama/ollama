/**
 * Workflow Templates
 *
 * Pre-defined workflow templates for common automation tasks
 * These can be triggered by natural language requests
 */

const workflowTemplates = {
    /**
     * Generic appointment booking workflow
     */
    appointmentBooking: {
        name: 'Appointment Booking',
        description: 'Book an appointment at a specified venue',
        triggers: [
            'book appointment',
            'book a haircut',
            'schedule appointment',
            'make reservation',
            'book salon',
            'book doctor',
            'book dentist'
        ],
        requiredParams: ['venue', 'date', 'time'],
        optionalParams: ['service', 'notes'],
        steps: [
            {
                type: 'tool_call',
                name: 'search_venue',
                tool: 'browser_navigate',
                params: { url: '{{venueUrl}}' },
                outputKey: 'venuePage',
                onError: 'stop'
            },
            {
                type: 'tool_call',
                name: 'analyze_page',
                tool: 'browser_get_text',
                params: {},
                outputKey: 'pageContent'
            },
            {
                type: 'agent_call',
                name: 'find_booking_section',
                agent: 'researcher',
                message: `Analyze this page content and identify how to book an appointment:

Page content: {{pageContent}}

I need to book: {{service}} on {{date}} at {{time}}

Identify:
1. The booking button/link selector
2. Any forms that need to be filled
3. Available time slots if visible

Return JSON with: { bookingSelector, formFields, availableSlots }`,
                outputKey: 'bookingAnalysis'
            },
            {
                type: 'tool_call',
                name: 'click_booking',
                tool: 'browser_click',
                params: { selector: '{{bookingAnalysis.bookingSelector}}' },
                onError: 'continue'
            },
            {
                type: 'tool_call',
                name: 'get_form_fields',
                tool: 'browser_get_forms',
                params: {},
                outputKey: 'formFields'
            },
            {
                type: 'agent_call',
                name: 'fill_form_strategy',
                agent: 'planner',
                message: `Plan how to fill this booking form:

Form fields: {{formFields}}

User wants to book: {{service}} on {{date}} at {{time}}
User email: {{userEmail}}
User name: {{userName}}

Return a JSON array of form filling actions: [{ selector, value }]`,
                outputKey: 'formStrategy'
            },
            {
                type: 'loop',
                name: 'fill_form_fields',
                items: '{{formStrategy}}',
                itemKey: 'field',
                step: {
                    type: 'tool_call',
                    name: 'fill_field',
                    tool: 'browser_fill',
                    params: {
                        selector: '{{field.selector}}',
                        value: '{{field.value}}'
                    }
                }
            },
            {
                type: 'tool_call',
                name: 'submit_booking',
                tool: 'browser_click',
                params: { selector: 'button[type="submit"], .submit-btn, #confirm-booking' },
                outputKey: 'submitResult'
            },
            {
                type: 'tool_call',
                name: 'capture_confirmation',
                tool: 'browser_get_text',
                params: {},
                outputKey: 'confirmationPage'
            }
        ]
    },

    /**
     * Web search and summarize workflow
     */
    webSearch: {
        name: 'Web Search',
        description: 'Search the web and summarize results',
        triggers: [
            'search for',
            'look up',
            'find information about',
            'what is',
            'research'
        ],
        requiredParams: ['query'],
        steps: [
            {
                type: 'tool_call',
                name: 'search',
                tool: 'web_search',
                params: { query: '{{query}}' },
                outputKey: 'searchResults'
            },
            {
                type: 'agent_call',
                name: 'summarize',
                agent: 'researcher',
                message: `Summarize these search results for: "{{query}}"

Results: {{searchResults}}

Provide a clear, concise summary with key points.`,
                outputKey: 'summary'
            }
        ]
    },

    /**
     * Email sending workflow
     */
    sendEmail: {
        name: 'Send Email',
        description: 'Compose and send an email',
        triggers: [
            'send email',
            'email to',
            'send a message to',
            'compose email'
        ],
        requiredParams: ['to', 'subject', 'body'],
        steps: [
            {
                type: 'agent_call',
                name: 'compose_email',
                agent: 'researcher',
                message: `Compose a professional email:

To: {{to}}
Subject: {{subject}}
User's request: {{body}}

Write the email body in a professional tone. Return just the email body text.`,
                outputKey: 'composedBody'
            },
            {
                type: 'tool_call',
                name: 'send_email',
                tool: 'send_email',
                params: {
                    to: '{{to}}',
                    subject: '{{subject}}',
                    body: '{{composedBody}}'
                },
                outputKey: 'sendResult'
            }
        ]
    },

    /**
     * Calendar event creation workflow
     */
    createCalendarEvent: {
        name: 'Create Calendar Event',
        description: 'Add an event to the calendar',
        triggers: [
            'add to calendar',
            'schedule',
            'create event',
            'set reminder',
            'add meeting'
        ],
        requiredParams: ['title', 'date', 'time'],
        optionalParams: ['duration', 'location', 'description'],
        steps: [
            {
                type: 'agent_call',
                name: 'parse_datetime',
                agent: 'planner',
                message: `Parse this date/time into ISO 8601 format:

Date: {{date}}
Time: {{time}}
Duration: {{duration}} (default 60 minutes if not specified)

Return JSON: { startTime: "ISO8601", endTime: "ISO8601" }`,
                outputKey: 'parsedTime'
            },
            {
                type: 'tool_call',
                name: 'create_event',
                tool: 'mcp_google_calendar_create_event',
                params: {
                    title: '{{title}}',
                    start: '{{parsedTime.startTime}}',
                    end: '{{parsedTime.endTime}}',
                    location: '{{location}}',
                    description: '{{description}}'
                },
                outputKey: 'calendarResult'
            }
        ]
    },

    /**
     * Price comparison workflow
     */
    priceComparison: {
        name: 'Price Comparison',
        description: 'Compare prices for a product across websites',
        triggers: [
            'compare prices',
            'find best price',
            'cheapest',
            'price check'
        ],
        requiredParams: ['product'],
        steps: [
            {
                type: 'parallel',
                name: 'search_multiple_sites',
                steps: [
                    {
                        type: 'tool_call',
                        name: 'search_amazon',
                        tool: 'browser_navigate',
                        params: { url: 'https://www.amazon.co.uk/s?k={{product}}' },
                        outputKey: 'amazonResults'
                    },
                    {
                        type: 'tool_call',
                        name: 'search_ebay',
                        tool: 'browser_navigate',
                        params: { url: 'https://www.ebay.co.uk/sch/i.html?_nkw={{product}}' },
                        outputKey: 'ebayResults'
                    }
                ]
            },
            {
                type: 'tool_call',
                name: 'extract_amazon_prices',
                tool: 'browser_extract',
                params: { selector: '.a-price-whole', sessionId: 'amazon' },
                outputKey: 'amazonPrices'
            },
            {
                type: 'agent_call',
                name: 'summarize_prices',
                agent: 'researcher',
                message: `Compare these prices for "{{product}}":

Amazon prices: {{amazonPrices}}
eBay prices: {{ebayPrices}}

Summarize the best deals found.`,
                outputKey: 'comparison'
            }
        ]
    },

    /**
     * Daily briefing workflow
     */
    dailyBriefing: {
        name: 'Daily Briefing',
        description: 'Get a summary of calendar, weather, and news',
        triggers: [
            'daily briefing',
            'morning summary',
            'what\'s happening today',
            'my day'
        ],
        steps: [
            {
                type: 'parallel',
                name: 'gather_info',
                steps: [
                    {
                        type: 'tool_call',
                        name: 'get_calendar',
                        tool: 'mcp_google_calendar_list_events',
                        params: { date: 'today' },
                        outputKey: 'calendarEvents'
                    },
                    {
                        type: 'tool_call',
                        name: 'get_weather',
                        tool: 'web_search',
                        params: { query: 'weather today {{location}}' },
                        outputKey: 'weather'
                    }
                ]
            },
            {
                type: 'agent_call',
                name: 'compose_briefing',
                agent: 'researcher',
                message: `Create a friendly daily briefing:

Calendar events: {{calendarEvents}}
Weather: {{weather}}

Format as a concise, friendly morning summary.`,
                outputKey: 'briefing'
            }
        ]
    }
};

/**
 * Workflow matcher - finds matching workflow from natural language
 */
function findMatchingWorkflow(userMessage) {
    const message = userMessage.toLowerCase();

    for (const [key, workflow] of Object.entries(workflowTemplates)) {
        for (const trigger of workflow.triggers || []) {
            if (message.includes(trigger.toLowerCase())) {
                return { key, workflow };
            }
        }
    }

    return null;
}

/**
 * Extract parameters from user message for a workflow
 */
function extractWorkflowParams(userMessage, workflow) {
    const params = {};

    // Common patterns for extraction
    const patterns = {
        date: /(?:on|for)\s+(\w+(?:\s+\d{1,2}(?:st|nd|rd|th)?)?|\d{1,2}\/\d{1,2}(?:\/\d{2,4})?|tomorrow|today|next\s+\w+)/i,
        time: /(?:at|@)\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)/i,
        email: /[\w.-]+@[\w.-]+\.\w+/,
        url: /https?:\/\/[^\s]+/,
        venue: /(?:at|to|from)\s+([^,]+?)(?:\s+(?:on|at|for)|\s*$)/i
    };

    for (const [param, pattern] of Object.entries(patterns)) {
        const match = userMessage.match(pattern);
        if (match) {
            params[param] = match[1] || match[0];
        }
    }

    return params;
}

module.exports = {
    workflowTemplates,
    findMatchingWorkflow,
    extractWorkflowParams
};
