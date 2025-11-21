/**
 * Telegram Bot Adapter
 *
 * Integrates Telegram messaging with the agent system
 * Supports commands, inline keyboards, and rich formatting
 */

const TelegramBot = require('node-telegram-bot-api');
const EventEmitter = require('events');

class TelegramAdapter extends EventEmitter {
    constructor(token, options = {}) {
        super();
        this.logger = options.logger || console;
        this.token = token;
        this.bot = null;
        this.options = options;
        this.commands = new Map();
        this.pendingCallbacks = new Map();  // For inline keyboard callbacks

        this.initialize();
    }

    /**
     * Initialize the Telegram bot
     */
    initialize() {
        if (!this.token) {
            this.logger.warn('Telegram bot token not provided, adapter disabled');
            return;
        }

        // Create bot instance
        this.bot = new TelegramBot(this.token, {
            polling: {
                autoStart: true,
                params: {
                    timeout: 30
                }
            }
        });

        this.setupHandlers();
        this.registerDefaultCommands();

        this.logger.info('Telegram bot initialized and polling');
    }

    /**
     * Set up message and event handlers
     */
    setupHandlers() {
        // Handle regular text messages
        this.bot.on('message', (msg) => {
            // Skip if it's a command (handled separately)
            if (msg.text && msg.text.startsWith('/')) {
                return;
            }

            if (msg.text) {
                this.emit('message', {
                    userId: msg.chat.id.toString(),
                    text: msg.text,
                    metadata: {
                        username: msg.from?.username,
                        firstName: msg.from?.first_name,
                        lastName: msg.from?.last_name,
                        chatType: msg.chat.type,
                        messageId: msg.message_id
                    }
                });
            }
        });

        // Handle callback queries (inline keyboard buttons)
        this.bot.on('callback_query', async (query) => {
            const callbackId = query.data;
            const callback = this.pendingCallbacks.get(callbackId);

            if (callback) {
                await callback(query);
                this.pendingCallbacks.delete(callbackId);
            }

            // Acknowledge the callback
            await this.bot.answerCallbackQuery(query.id);
        });

        // Handle photos
        this.bot.on('photo', (msg) => {
            const photo = msg.photo[msg.photo.length - 1]; // Get highest resolution
            this.emit('message', {
                userId: msg.chat.id.toString(),
                text: msg.caption || '[Photo received]',
                metadata: {
                    type: 'photo',
                    fileId: photo.file_id,
                    width: photo.width,
                    height: photo.height
                }
            });
        });

        // Handle documents
        this.bot.on('document', (msg) => {
            this.emit('message', {
                userId: msg.chat.id.toString(),
                text: msg.caption || `[Document: ${msg.document.file_name}]`,
                metadata: {
                    type: 'document',
                    fileId: msg.document.file_id,
                    fileName: msg.document.file_name,
                    mimeType: msg.document.mime_type
                }
            });
        });

        // Handle voice messages
        this.bot.on('voice', (msg) => {
            this.emit('message', {
                userId: msg.chat.id.toString(),
                text: '[Voice message received - transcription not available]',
                metadata: {
                    type: 'voice',
                    fileId: msg.voice.file_id,
                    duration: msg.voice.duration
                }
            });
        });

        // Handle polling errors
        this.bot.on('polling_error', (error) => {
            this.logger.error('Telegram polling error:', error.message);
            this.emit('error', error);
        });

        // Handle webhook errors
        this.bot.on('webhook_error', (error) => {
            this.logger.error('Telegram webhook error:', error.message);
            this.emit('error', error);
        });
    }

    /**
     * Register default bot commands
     */
    registerDefaultCommands() {
        // /start command
        this.registerCommand('start', async (msg) => {
            const welcomeMessage = `
Hello ${msg.from?.first_name || 'there'}! I'm your personal AI assistant powered by Ollama.

I can help you with:
- Booking appointments
- Managing your calendar
- Sending emails
- Web research and browsing
- General questions and tasks

Just send me a message describing what you need!

Use /help to see available commands.
            `.trim();

            await this.sendMessage(msg.chat.id, welcomeMessage);
        });

        // /help command
        this.registerCommand('help', async (msg) => {
            const helpMessage = `
*Available Commands*

/start - Welcome message
/help - Show this help
/status - Check system status
/clear - Clear conversation history
/settings - View/change settings

*Tips*
- Just type naturally - I understand context
- I can browse websites and fill forms
- I can manage your calendar and send emails
- Use specific details for best results

*Example requests:*
- "Book a haircut for Saturday"
- "What's on my calendar tomorrow?"
- "Search for the best pizza place nearby"
            `.trim();

            await this.sendMessage(msg.chat.id, helpMessage, { parse_mode: 'Markdown' });
        });

        // /status command
        this.registerCommand('status', async (msg) => {
            const statusMessage = `
*System Status*

Bot: Online
Ollama: Checking...
MCP Servers: Checking...

Use this to verify the system is running correctly.
            `.trim();

            await this.sendMessage(msg.chat.id, statusMessage, { parse_mode: 'Markdown' });

            // Emit status check event
            this.emit('statusCheck', { userId: msg.chat.id.toString() });
        });

        // /clear command
        this.registerCommand('clear', async (msg) => {
            this.emit('clearHistory', {
                userId: msg.chat.id.toString(),
                platform: 'telegram'
            });
            await this.sendMessage(msg.chat.id, 'Conversation history cleared. Starting fresh!');
        });

        // Set commands in Telegram UI
        this.bot.setMyCommands([
            { command: 'start', description: 'Start the bot' },
            { command: 'help', description: 'Show help message' },
            { command: 'status', description: 'Check system status' },
            { command: 'clear', description: 'Clear conversation history' },
            { command: 'settings', description: 'View settings' }
        ]).catch(err => this.logger.warn('Failed to set bot commands:', err.message));
    }

    /**
     * Register a custom command
     */
    registerCommand(command, handler) {
        this.commands.set(command, handler);
        this.bot.onText(new RegExp(`^/${command}(?:@\\w+)?(?:\\s+(.*))?$`), async (msg, match) => {
            try {
                await handler(msg, match ? match[1] : null);
            } catch (error) {
                this.logger.error(`Command /${command} error:`, error);
                await this.sendMessage(msg.chat.id, `Error executing command: ${error.message}`);
            }
        });
    }

    /**
     * Send a message to a user
     */
    async sendMessage(userId, text, options = {}) {
        if (!this.bot) {
            throw new Error('Telegram bot not initialized');
        }

        // Split long messages (Telegram limit is 4096 chars)
        const chunks = this.splitMessage(text, 4000);

        for (let i = 0; i < chunks.length; i++) {
            const chunk = chunks[i];
            const isLastChunk = i === chunks.length - 1;

            await this.bot.sendMessage(userId, chunk, {
                parse_mode: options.parse_mode || 'Markdown',
                disable_web_page_preview: options.disable_preview !== false,
                reply_markup: isLastChunk ? options.reply_markup : undefined,
                ...options
            }).catch(async (err) => {
                // If Markdown fails, try without formatting
                if (err.message.includes('parse')) {
                    await this.bot.sendMessage(userId, chunk, {
                        disable_web_page_preview: true
                    });
                } else {
                    throw err;
                }
            });
        }
    }

    /**
     * Send typing indicator
     */
    async sendTypingIndicator(userId) {
        if (!this.bot) return;
        await this.bot.sendChatAction(userId, 'typing');
    }

    /**
     * Send a message with inline keyboard
     */
    async sendWithKeyboard(userId, text, buttons) {
        if (!this.bot) return;

        // Convert button format to Telegram format
        const keyboard = buttons.map(row =>
            row.map(btn => {
                const callbackId = `btn_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

                if (btn.callback) {
                    this.pendingCallbacks.set(callbackId, btn.callback);
                    return { text: btn.text, callback_data: callbackId };
                } else if (btn.url) {
                    return { text: btn.text, url: btn.url };
                }
                return { text: btn.text, callback_data: callbackId };
            })
        );

        await this.bot.sendMessage(userId, text, {
            parse_mode: 'Markdown',
            reply_markup: {
                inline_keyboard: keyboard
            }
        });
    }

    /**
     * Send a photo
     */
    async sendPhoto(userId, photo, caption) {
        if (!this.bot) return;
        await this.bot.sendPhoto(userId, photo, { caption });
    }

    /**
     * Send a document
     */
    async sendDocument(userId, document, caption) {
        if (!this.bot) return;
        await this.bot.sendDocument(userId, document, { caption });
    }

    /**
     * Edit a previous message
     */
    async editMessage(userId, messageId, newText) {
        if (!this.bot) return;
        await this.bot.editMessageText(newText, {
            chat_id: userId,
            message_id: messageId,
            parse_mode: 'Markdown'
        });
    }

    /**
     * Delete a message
     */
    async deleteMessage(userId, messageId) {
        if (!this.bot) return;
        await this.bot.deleteMessage(userId, messageId);
    }

    /**
     * Get user profile info
     */
    async getUserInfo(userId) {
        if (!this.bot) return null;
        try {
            const chat = await this.bot.getChat(userId);
            return {
                id: chat.id,
                username: chat.username,
                firstName: chat.first_name,
                lastName: chat.last_name,
                bio: chat.bio,
                photo: chat.photo
            };
        } catch (error) {
            this.logger.error('Failed to get user info:', error);
            return null;
        }
    }

    /**
     * Download a file (photo, document, voice)
     */
    async downloadFile(fileId) {
        if (!this.bot) return null;
        const fileLink = await this.bot.getFileLink(fileId);
        return fileLink;
    }

    /**
     * Split message into chunks
     */
    splitMessage(text, maxLength) {
        if (text.length <= maxLength) {
            return [text];
        }

        const chunks = [];
        let remaining = text;

        while (remaining.length > 0) {
            let chunk = remaining.slice(0, maxLength);

            // Try to break at natural points
            if (remaining.length > maxLength) {
                // Try paragraph break
                let breakPoint = chunk.lastIndexOf('\n\n');

                // Try line break
                if (breakPoint < maxLength / 2) {
                    breakPoint = chunk.lastIndexOf('\n');
                }

                // Try sentence break
                if (breakPoint < maxLength / 2) {
                    breakPoint = Math.max(
                        chunk.lastIndexOf('. '),
                        chunk.lastIndexOf('! '),
                        chunk.lastIndexOf('? ')
                    );
                }

                // Try word break
                if (breakPoint < maxLength / 2) {
                    breakPoint = chunk.lastIndexOf(' ');
                }

                if (breakPoint > 0) {
                    chunk = chunk.slice(0, breakPoint + 1);
                }
            }

            chunks.push(chunk.trim());
            remaining = remaining.slice(chunk.length).trim();
        }

        return chunks;
    }

    /**
     * Check if bot is running
     */
    isRunning() {
        return this.bot !== null;
    }

    /**
     * Get bot info
     */
    async getBotInfo() {
        if (!this.bot) return null;
        return await this.bot.getMe();
    }

    /**
     * Shutdown the bot
     */
    async shutdown() {
        if (this.bot) {
            await this.bot.stopPolling();
            this.bot = null;
            this.logger.info('Telegram bot shutdown complete');
        }
    }
}

module.exports = TelegramAdapter;
