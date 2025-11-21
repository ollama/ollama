/**
 * Personal Knowledge Base
 *
 * A RAG (Retrieval Augmented Generation) system that:
 * 1. Indexes your personal files, emails, notes
 * 2. Stores facts learned from conversations
 * 3. Retrieves relevant context for each query
 *
 * This makes the agent truly understand YOU.
 */

const fs = require('fs').promises;
const path = require('path');
const crypto = require('crypto');

class PersonalKnowledgeBase {
    constructor(options = {}) {
        this.logger = options.logger || console;
        this.dataDir = options.dataDir || path.join(__dirname, '../data/knowledge');
        this.callOllama = options.callOllama;

        // In-memory stores (persisted to disk)
        this.documents = new Map();      // docId → { content, metadata, embedding }
        this.facts = new Map();          // factId → { fact, source, confidence, timestamp }
        this.conversations = [];         // Past conversation summaries
        this.userProfile = {};           // Structured profile data

        // Simple text-based search (upgrade to vector DB for production)
        this.searchIndex = new Map();    // word → [docIds]
    }

    /**
     * Initialize the knowledge base
     */
    async initialize() {
        await fs.mkdir(this.dataDir, { recursive: true });

        // Load existing data
        await this.loadDocuments();
        await this.loadFacts();
        await this.loadProfile();

        this.logger.info(`Knowledge base initialized: ${this.documents.size} documents, ${this.facts.size} facts`);
    }

    // ==========================================
    // DOCUMENT INDEXING
    // ==========================================

    /**
     * Index a single file
     */
    async indexFile(filePath) {
        try {
            const content = await fs.readFile(filePath, 'utf8');
            const stats = await fs.stat(filePath);

            const docId = this.generateId(filePath);
            const chunks = this.chunkText(content);

            const document = {
                id: docId,
                path: filePath,
                filename: path.basename(filePath),
                content: content.substring(0, 50000), // Limit size
                chunks,
                metadata: {
                    size: stats.size,
                    modified: stats.mtime,
                    indexed: new Date().toISOString(),
                    type: path.extname(filePath)
                }
            };

            this.documents.set(docId, document);
            this.indexForSearch(docId, content);

            this.logger.info(`Indexed: ${filePath}`);
            return docId;

        } catch (error) {
            this.logger.error(`Failed to index ${filePath}:`, error.message);
            return null;
        }
    }

    /**
     * Index a directory of files
     */
    async indexDirectory(dirPath, extensions = ['.txt', '.md', '.json', '.csv', '.html']) {
        const indexed = [];

        async function* walkDir(dir) {
            const files = await fs.readdir(dir, { withFileTypes: true });
            for (const file of files) {
                const filePath = path.join(dir, file.name);
                if (file.isDirectory()) {
                    yield* walkDir(filePath);
                } else if (extensions.some(ext => file.name.endsWith(ext))) {
                    yield filePath;
                }
            }
        }

        for await (const filePath of walkDir(dirPath)) {
            const docId = await this.indexFile(filePath);
            if (docId) indexed.push(docId);
        }

        await this.saveDocuments();
        this.logger.info(`Indexed ${indexed.length} files from ${dirPath}`);
        return indexed;
    }

    /**
     * Index emails from exported file (mbox, json, or csv)
     */
    async indexEmails(emailsPath) {
        try {
            const content = await fs.readFile(emailsPath, 'utf8');
            let emails = [];

            // Try to parse as JSON first
            try {
                emails = JSON.parse(content);
            } catch {
                // Try as line-separated JSON
                emails = content.split('\n')
                    .filter(line => line.trim())
                    .map(line => {
                        try { return JSON.parse(line); }
                        catch { return null; }
                    })
                    .filter(Boolean);
            }

            let indexed = 0;
            for (const email of emails) {
                const docId = this.generateId(`email_${email.id || email.subject || indexed}`);

                const document = {
                    id: docId,
                    type: 'email',
                    subject: email.subject || 'No subject',
                    from: email.from || email.sender,
                    to: email.to || email.recipient,
                    date: email.date,
                    content: email.body || email.content || email.text,
                    metadata: {
                        type: 'email',
                        indexed: new Date().toISOString()
                    }
                };

                this.documents.set(docId, document);
                this.indexForSearch(docId, `${document.subject} ${document.content}`);
                indexed++;
            }

            await this.saveDocuments();
            this.logger.info(`Indexed ${indexed} emails`);
            return indexed;

        } catch (error) {
            this.logger.error('Failed to index emails:', error.message);
            return 0;
        }
    }

    /**
     * Index a piece of text directly (notes, clipboard, etc.)
     */
    async indexText(text, metadata = {}) {
        const docId = this.generateId(text.substring(0, 100));

        const document = {
            id: docId,
            type: metadata.type || 'note',
            title: metadata.title || 'Untitled',
            content: text,
            chunks: this.chunkText(text),
            metadata: {
                ...metadata,
                indexed: new Date().toISOString()
            }
        };

        this.documents.set(docId, document);
        this.indexForSearch(docId, text);
        await this.saveDocuments();

        return docId;
    }

    // ==========================================
    // FACT STORAGE (Learned from conversations)
    // ==========================================

    /**
     * Store a fact learned about the user
     */
    async storeFact(fact, options = {}) {
        const factId = this.generateId(fact);

        this.facts.set(factId, {
            id: factId,
            fact,
            category: options.category || 'general',
            source: options.source || 'conversation',
            confidence: options.confidence || 0.8,
            timestamp: new Date().toISOString(),
            lastUsed: null
        });

        await this.saveFacts();
        this.logger.info(`Stored fact: ${fact.substring(0, 50)}...`);
        return factId;
    }

    /**
     * Extract and store facts from a conversation
     */
    async extractFactsFromConversation(messages) {
        if (!this.callOllama) return [];

        const conversationText = messages
            .map(m => `${m.role}: ${m.content}`)
            .join('\n');

        const prompt = `Extract personal facts about the user from this conversation.

Conversation:
${conversationText}

Extract facts like:
- Name, location, job
- Preferences (food, music, etc.)
- Schedule, routines
- Relationships (mentioned people)
- Goals, interests
- Important dates

Return JSON array of facts:
[
  { "fact": "User's name is John", "category": "identity", "confidence": 0.95 },
  { "fact": "User lives in London", "category": "location", "confidence": 0.9 }
]

Only include facts explicitly stated or strongly implied. Return [] if no facts found.`;

        try {
            const response = await this.callOllama('researcher', prompt);
            const jsonMatch = response.match(/\[[\s\S]*\]/);

            if (jsonMatch) {
                const facts = JSON.parse(jsonMatch[0]);
                for (const factData of facts) {
                    await this.storeFact(factData.fact, {
                        category: factData.category,
                        confidence: factData.confidence,
                        source: 'conversation_extraction'
                    });
                }
                return facts;
            }
        } catch (error) {
            this.logger.warn('Failed to extract facts:', error.message);
        }

        return [];
    }

    /**
     * Get facts by category
     */
    getFactsByCategory(category) {
        return Array.from(this.facts.values())
            .filter(f => f.category === category)
            .sort((a, b) => b.confidence - a.confidence);
    }

    /**
     * Get all facts as formatted text
     */
    getFactsSummary() {
        const factsByCategory = {};

        for (const fact of this.facts.values()) {
            if (!factsByCategory[fact.category]) {
                factsByCategory[fact.category] = [];
            }
            factsByCategory[fact.category].push(fact.fact);
        }

        let summary = 'WHAT I KNOW ABOUT YOU:\n\n';
        for (const [category, facts] of Object.entries(factsByCategory)) {
            summary += `${category.toUpperCase()}:\n`;
            facts.forEach(f => summary += `- ${f}\n`);
            summary += '\n';
        }

        return summary;
    }

    // ==========================================
    // SEARCH & RETRIEVAL
    // ==========================================

    /**
     * Search documents for relevant context
     */
    search(query, limit = 5) {
        const queryWords = this.tokenize(query.toLowerCase());
        const scores = new Map();

        // Score each document
        for (const word of queryWords) {
            const docIds = this.searchIndex.get(word) || [];
            for (const docId of docIds) {
                scores.set(docId, (scores.get(docId) || 0) + 1);
            }
        }

        // Sort by score and return top results
        return Array.from(scores.entries())
            .sort((a, b) => b[1] - a[1])
            .slice(0, limit)
            .map(([docId, score]) => ({
                document: this.documents.get(docId),
                score
            }));
    }

    /**
     * Search facts for relevant context
     */
    searchFacts(query, limit = 10) {
        const queryLower = query.toLowerCase();
        const queryWords = this.tokenize(queryLower);

        return Array.from(this.facts.values())
            .map(fact => {
                const factLower = fact.fact.toLowerCase();
                let score = 0;

                // Score based on word matches
                for (const word of queryWords) {
                    if (factLower.includes(word)) score += 1;
                }

                // Boost by confidence
                score *= fact.confidence;

                return { fact, score };
            })
            .filter(r => r.score > 0)
            .sort((a, b) => b.score - a.score)
            .slice(0, limit)
            .map(r => r.fact);
    }

    /**
     * Get relevant context for a query
     * This is the main method used by the agent
     */
    async getRelevantContext(query, options = {}) {
        const maxTokens = options.maxTokens || 2000;
        let context = '';

        // 1. Search relevant documents
        const docs = this.search(query, 3);
        if (docs.length > 0) {
            context += '=== RELEVANT DOCUMENTS ===\n\n';
            for (const { document } of docs) {
                const excerpt = document.content?.substring(0, 500) || '';
                context += `[${document.filename || document.title || 'Document'}]\n${excerpt}\n\n`;
            }
        }

        // 2. Get relevant facts
        const facts = this.searchFacts(query, 5);
        if (facts.length > 0) {
            context += '=== KNOWN FACTS ===\n';
            facts.forEach(f => context += `- ${f.fact}\n`);
            context += '\n';
        }

        // 3. Add user profile summary
        if (Object.keys(this.userProfile).length > 0) {
            context += '=== USER PROFILE ===\n';
            context += JSON.stringify(this.userProfile, null, 2);
            context += '\n\n';
        }

        // Truncate if too long
        if (context.length > maxTokens * 4) {
            context = context.substring(0, maxTokens * 4);
        }

        return context;
    }

    // ==========================================
    // USER PROFILE
    // ==========================================

    /**
     * Update user profile
     */
    async updateProfile(updates) {
        this.userProfile = {
            ...this.userProfile,
            ...updates,
            lastUpdated: new Date().toISOString()
        };
        await this.saveProfile();
    }

    /**
     * Set a specific profile field
     */
    async setProfileField(field, value) {
        this.userProfile[field] = value;
        this.userProfile.lastUpdated = new Date().toISOString();
        await this.saveProfile();
    }

    /**
     * Get profile field
     */
    getProfileField(field) {
        return this.userProfile[field];
    }

    // ==========================================
    // HELPER METHODS
    // ==========================================

    /**
     * Generate a unique ID
     */
    generateId(input) {
        return crypto.createHash('md5').update(input + Date.now()).digest('hex').substring(0, 12);
    }

    /**
     * Tokenize text for search
     */
    tokenize(text) {
        return text
            .toLowerCase()
            .replace(/[^\w\s]/g, ' ')
            .split(/\s+/)
            .filter(word => word.length > 2);
    }

    /**
     * Build search index for a document
     */
    indexForSearch(docId, content) {
        const words = this.tokenize(content);
        const uniqueWords = [...new Set(words)];

        for (const word of uniqueWords) {
            if (!this.searchIndex.has(word)) {
                this.searchIndex.set(word, []);
            }
            this.searchIndex.get(word).push(docId);
        }
    }

    /**
     * Chunk text into smaller pieces
     */
    chunkText(text, chunkSize = 1000, overlap = 200) {
        const chunks = [];
        let start = 0;

        while (start < text.length) {
            const end = Math.min(start + chunkSize, text.length);
            chunks.push({
                text: text.substring(start, end),
                start,
                end
            });
            start += chunkSize - overlap;
        }

        return chunks;
    }

    // ==========================================
    // PERSISTENCE
    // ==========================================

    async saveDocuments() {
        const data = Object.fromEntries(this.documents);
        await fs.writeFile(
            path.join(this.dataDir, 'documents.json'),
            JSON.stringify(data, null, 2)
        );
    }

    async loadDocuments() {
        try {
            const data = JSON.parse(
                await fs.readFile(path.join(this.dataDir, 'documents.json'), 'utf8')
            );
            this.documents = new Map(Object.entries(data));

            // Rebuild search index
            for (const [docId, doc] of this.documents) {
                this.indexForSearch(docId, doc.content || '');
            }
        } catch (error) {
            if (error.code !== 'ENOENT') {
                this.logger.warn('Failed to load documents:', error.message);
            }
        }
    }

    async saveFacts() {
        const data = Object.fromEntries(this.facts);
        await fs.writeFile(
            path.join(this.dataDir, 'facts.json'),
            JSON.stringify(data, null, 2)
        );
    }

    async loadFacts() {
        try {
            const data = JSON.parse(
                await fs.readFile(path.join(this.dataDir, 'facts.json'), 'utf8')
            );
            this.facts = new Map(Object.entries(data));
        } catch (error) {
            if (error.code !== 'ENOENT') {
                this.logger.warn('Failed to load facts:', error.message);
            }
        }
    }

    async saveProfile() {
        await fs.writeFile(
            path.join(this.dataDir, 'profile.json'),
            JSON.stringify(this.userProfile, null, 2)
        );
    }

    async loadProfile() {
        try {
            this.userProfile = JSON.parse(
                await fs.readFile(path.join(this.dataDir, 'profile.json'), 'utf8')
            );
        } catch (error) {
            if (error.code !== 'ENOENT') {
                this.logger.warn('Failed to load profile:', error.message);
            }
            this.userProfile = {};
        }
    }

    /**
     * Get statistics about the knowledge base
     */
    getStats() {
        return {
            documents: this.documents.size,
            facts: this.facts.size,
            searchIndexSize: this.searchIndex.size,
            profileFields: Object.keys(this.userProfile).length
        };
    }
}

module.exports = PersonalKnowledgeBase;
