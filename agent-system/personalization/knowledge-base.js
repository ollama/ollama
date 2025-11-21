/**
 * Personal Knowledge Base (Production-Ready)
 *
 * A RAG (Retrieval Augmented Generation) system with:
 * 1. Vector embeddings via Ollama (nomic-embed-text)
 * 2. SQLite persistence for facts and metadata
 * 3. Semantic search with cosine similarity
 * 4. Hierarchical document summarization
 */

const fs = require('fs').promises;
const path = require('path');
const crypto = require('crypto');

// SQLite for persistence
let Database;
try {
    Database = require('better-sqlite3');
} catch (e) {
    // Fallback to in-memory if better-sqlite3 not available
    Database = null;
}

class PersonalKnowledgeBase {
    constructor(options = {}) {
        this.logger = options.logger || console;
        this.dataDir = options.dataDir || path.join(__dirname, '../data/knowledge');
        this.callOllama = options.callOllama;
        this.ollamaApi = options.ollamaApi || process.env.OLLAMA_API || 'http://localhost:11434';

        // Embedding model - nomic-embed-text is optimized for RAG
        this.embeddingModel = options.embeddingModel || 'nomic-embed-text';

        // SQLite database (persistent)
        this.db = null;
        this.dbPath = path.join(this.dataDir, 'knowledge.db');

        // In-memory caches for fast access
        this.embeddingsCache = new Map();  // docId → embedding vector
        this.documentsCache = new Map();   // docId → document metadata
        this.factsCache = new Map();       // factId → fact data

        // User profile
        this.userProfile = {};

        // Configuration
        this.config = {
            maxContextTokens: options.maxContextTokens || 2000,
            maxDocsPerQuery: options.maxDocsPerQuery || 5,
            maxFactsPerQuery: options.maxFactsPerQuery || 10,
            similarityThreshold: options.similarityThreshold || 0.5,
            chunkSize: options.chunkSize || 500,
            chunkOverlap: options.chunkOverlap || 100
        };
    }

    /**
     * Initialize the knowledge base
     */
    async initialize() {
        await fs.mkdir(this.dataDir, { recursive: true });

        // Initialize SQLite database
        await this.initializeDatabase();

        // Load caches from database
        await this.loadCaches();

        // Load user profile
        await this.loadProfile();

        const stats = this.getStats();
        this.logger.info(`Knowledge base initialized: ${stats.documents} documents, ${stats.facts} facts, ${stats.embeddingsLoaded} embeddings cached`);
    }

    /**
     * Initialize SQLite database with schema
     */
    async initializeDatabase() {
        if (!Database) {
            this.logger.warn('better-sqlite3 not available, using JSON file fallback');
            await this.initializeJsonFallback();
            return;
        }

        try {
            this.db = new Database(this.dbPath);

            // Create tables
            this.db.exec(`
                -- Documents table
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    type TEXT DEFAULT 'file',
                    path TEXT,
                    filename TEXT,
                    title TEXT,
                    content TEXT,
                    summary TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Document chunks for fine-grained retrieval
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT,
                    chunk_index INTEGER,
                    content TEXT,
                    start_pos INTEGER,
                    end_pos INTEGER,
                    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
                );

                -- Embeddings table (stored as binary blobs)
                CREATE TABLE IF NOT EXISTS embeddings (
                    id TEXT PRIMARY KEY,
                    type TEXT DEFAULT 'document',  -- document, chunk, fact
                    target_id TEXT,
                    embedding BLOB,
                    model TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Facts table
                CREATE TABLE IF NOT EXISTS facts (
                    id TEXT PRIMARY KEY,
                    fact TEXT NOT NULL,
                    category TEXT DEFAULT 'general',
                    source TEXT DEFAULT 'conversation',
                    confidence REAL DEFAULT 0.8,
                    times_used INTEGER DEFAULT 0,
                    last_used TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Conversation summaries for long-term memory
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    summary TEXT,
                    key_topics TEXT,
                    facts_extracted TEXT,
                    message_count INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Search index for fast text search (backup for embedding failures)
                CREATE VIRTUAL TABLE IF NOT EXISTS search_index USING fts5(
                    doc_id,
                    content,
                    tokenize='porter'
                );

                -- Create indexes
                CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
                CREATE INDEX IF NOT EXISTS idx_embeddings_target ON embeddings(target_id);
                CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
                CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);
            `);

            this.logger.info('SQLite database initialized');
        } catch (error) {
            this.logger.error('Failed to initialize SQLite:', error.message);
            await this.initializeJsonFallback();
        }
    }

    /**
     * Fallback to JSON files if SQLite unavailable
     */
    async initializeJsonFallback() {
        this.useJsonFallback = true;
        this.jsonFiles = {
            documents: path.join(this.dataDir, 'documents.json'),
            facts: path.join(this.dataDir, 'facts.json'),
            embeddings: path.join(this.dataDir, 'embeddings.json')
        };
    }

    /**
     * Load caches from database
     */
    async loadCaches() {
        if (this.useJsonFallback) {
            await this.loadJsonCaches();
            return;
        }

        try {
            // Load documents
            const docs = this.db.prepare('SELECT * FROM documents').all();
            for (const doc of docs) {
                this.documentsCache.set(doc.id, {
                    ...doc,
                    metadata: doc.metadata ? JSON.parse(doc.metadata) : {}
                });
            }

            // Load facts
            const facts = this.db.prepare('SELECT * FROM facts').all();
            for (const fact of facts) {
                this.factsCache.set(fact.id, fact);
            }

            // Load embeddings into memory (for fast similarity search)
            const embeddings = this.db.prepare('SELECT * FROM embeddings').all();
            for (const emb of embeddings) {
                if (emb.embedding) {
                    // Convert blob back to Float32Array
                    const buffer = emb.embedding;
                    const vector = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.length / 4);
                    this.embeddingsCache.set(emb.target_id, Array.from(vector));
                }
            }
        } catch (error) {
            this.logger.error('Failed to load caches:', error.message);
        }
    }

    /**
     * Load from JSON files (fallback)
     */
    async loadJsonCaches() {
        try {
            const docsData = await fs.readFile(this.jsonFiles.documents, 'utf8').catch(() => '{}');
            const docs = JSON.parse(docsData);
            this.documentsCache = new Map(Object.entries(docs));

            const factsData = await fs.readFile(this.jsonFiles.facts, 'utf8').catch(() => '{}');
            const facts = JSON.parse(factsData);
            this.factsCache = new Map(Object.entries(facts));

            const embData = await fs.readFile(this.jsonFiles.embeddings, 'utf8').catch(() => '{}');
            const embeddings = JSON.parse(embData);
            this.embeddingsCache = new Map(Object.entries(embeddings));
        } catch (error) {
            this.logger.warn('Failed to load JSON caches:', error.message);
        }
    }

    // ==========================================
    // EMBEDDINGS
    // ==========================================

    /**
     * Generate embedding for text using Ollama
     */
    async generateEmbedding(text) {
        try {
            const response = await fetch(`${this.ollamaApi}/api/embeddings`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: this.embeddingModel,
                    prompt: text.substring(0, 8000) // Limit text length
                })
            });

            if (!response.ok) {
                throw new Error(`Embedding request failed: ${response.status}`);
            }

            const data = await response.json();
            return data.embedding;
        } catch (error) {
            this.logger.warn(`Failed to generate embedding: ${error.message}`);
            return null;
        }
    }

    /**
     * Calculate cosine similarity between two vectors
     */
    cosineSimilarity(a, b) {
        if (!a || !b || a.length !== b.length) return 0;

        let dotProduct = 0;
        let normA = 0;
        let normB = 0;

        for (let i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
        return magnitude === 0 ? 0 : dotProduct / magnitude;
    }

    /**
     * Store embedding in database
     */
    async storeEmbedding(targetId, embedding, type = 'document') {
        if (!embedding) return;

        this.embeddingsCache.set(targetId, embedding);

        if (this.db) {
            try {
                // Convert to binary buffer for storage
                const buffer = Buffer.from(new Float32Array(embedding).buffer);

                this.db.prepare(`
                    INSERT OR REPLACE INTO embeddings (id, type, target_id, embedding, model)
                    VALUES (?, ?, ?, ?, ?)
                `).run(
                    `emb_${targetId}`,
                    type,
                    targetId,
                    buffer,
                    this.embeddingModel
                );
            } catch (error) {
                this.logger.warn('Failed to store embedding:', error.message);
            }
        } else if (this.useJsonFallback) {
            await this.saveJsonEmbeddings();
        }
    }

    // ==========================================
    // DOCUMENT INDEXING
    // ==========================================

    /**
     * Index a single file with embeddings
     */
    async indexFile(filePath) {
        try {
            const content = await fs.readFile(filePath, 'utf8');
            const stats = await fs.stat(filePath);

            const docId = this.generateId(filePath);
            const chunks = this.chunkText(content);

            // Generate summary for long documents
            let summary = null;
            if (content.length > 2000 && this.callOllama) {
                summary = await this.summarizeDocument(content.substring(0, 10000));
            }

            const document = {
                id: docId,
                type: 'file',
                path: filePath,
                filename: path.basename(filePath),
                content: content.substring(0, 50000),
                summary,
                metadata: JSON.stringify({
                    size: stats.size,
                    modified: stats.mtime,
                    indexed: new Date().toISOString(),
                    fileType: path.extname(filePath),
                    chunkCount: chunks.length
                })
            };

            // Store document
            await this.storeDocument(document);

            // Generate and store embedding for document (use summary if available)
            const textForEmbedding = summary || content.substring(0, 2000);
            const embedding = await this.generateEmbedding(textForEmbedding);
            await this.storeEmbedding(docId, embedding, 'document');

            // Index chunks for fine-grained retrieval
            await this.indexChunks(docId, chunks);

            this.logger.info(`Indexed: ${filePath} (${chunks.length} chunks)`);
            return docId;

        } catch (error) {
            this.logger.error(`Failed to index ${filePath}:`, error.message);
            return null;
        }
    }

    /**
     * Index document chunks
     */
    async indexChunks(docId, chunks) {
        for (let i = 0; i < chunks.length; i++) {
            const chunk = chunks[i];
            const chunkId = `${docId}_chunk_${i}`;

            if (this.db) {
                this.db.prepare(`
                    INSERT OR REPLACE INTO chunks (id, doc_id, chunk_index, content, start_pos, end_pos)
                    VALUES (?, ?, ?, ?, ?, ?)
                `).run(chunkId, docId, i, chunk.text, chunk.start, chunk.end);
            }

            // Generate embedding for each chunk (async, don't await all)
            if (i < 10) { // Limit embedding generation for large docs
                const embedding = await this.generateEmbedding(chunk.text);
                await this.storeEmbedding(chunkId, embedding, 'chunk');
            }
        }
    }

    /**
     * Store document in database
     */
    async storeDocument(document) {
        this.documentsCache.set(document.id, {
            ...document,
            metadata: typeof document.metadata === 'string' ? JSON.parse(document.metadata) : document.metadata
        });

        if (this.db) {
            this.db.prepare(`
                INSERT OR REPLACE INTO documents (id, type, path, filename, title, content, summary, metadata, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            `).run(
                document.id,
                document.type,
                document.path,
                document.filename,
                document.title,
                document.content,
                document.summary,
                document.metadata
            );

            // Add to FTS search index
            try {
                this.db.prepare(`
                    INSERT OR REPLACE INTO search_index (doc_id, content)
                    VALUES (?, ?)
                `).run(document.id, document.content?.substring(0, 10000));
            } catch (e) {
                // FTS insert may fail, ignore
            }
        } else if (this.useJsonFallback) {
            await this.saveJsonDocuments();
        }
    }

    /**
     * Summarize a long document
     */
    async summarizeDocument(content) {
        if (!this.callOllama) return null;

        try {
            const prompt = `Summarize this document in 2-3 sentences, focusing on key facts and topics:\n\n${content.substring(0, 5000)}`;
            return await this.callOllama('researcher', prompt);
        } catch (error) {
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
                if (file.isDirectory() && !file.name.startsWith('.')) {
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

        this.logger.info(`Indexed ${indexed.length} files from ${dirPath}`);
        return indexed;
    }

    /**
     * Index emails from exported file
     */
    async indexEmails(emailsPath) {
        try {
            const content = await fs.readFile(emailsPath, 'utf8');
            let emails = [];

            try {
                emails = JSON.parse(content);
            } catch {
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
                const emailContent = `Subject: ${email.subject || 'No subject'}\nFrom: ${email.from || email.sender}\nTo: ${email.to || email.recipient}\nDate: ${email.date}\n\n${email.body || email.content || email.text}`;

                const document = {
                    id: docId,
                    type: 'email',
                    title: email.subject || 'No subject',
                    content: emailContent,
                    metadata: JSON.stringify({
                        type: 'email',
                        from: email.from || email.sender,
                        to: email.to || email.recipient,
                        date: email.date,
                        indexed: new Date().toISOString()
                    })
                };

                await this.storeDocument(document);

                // Generate embedding
                const embedding = await this.generateEmbedding(emailContent.substring(0, 2000));
                await this.storeEmbedding(docId, embedding, 'document');

                indexed++;
            }

            this.logger.info(`Indexed ${indexed} emails`);
            return indexed;

        } catch (error) {
            this.logger.error('Failed to index emails:', error.message);
            return 0;
        }
    }

    /**
     * Index text directly
     */
    async indexText(text, metadata = {}) {
        const docId = this.generateId(text.substring(0, 100));

        const document = {
            id: docId,
            type: metadata.type || 'note',
            title: metadata.title || 'Untitled',
            content: text,
            metadata: JSON.stringify({
                ...metadata,
                indexed: new Date().toISOString()
            })
        };

        await this.storeDocument(document);

        const embedding = await this.generateEmbedding(text.substring(0, 2000));
        await this.storeEmbedding(docId, embedding, 'document');

        return docId;
    }

    // ==========================================
    // FACT STORAGE
    // ==========================================

    /**
     * Store a fact with embedding
     */
    async storeFact(fact, options = {}) {
        const factId = this.generateId(fact);

        const factData = {
            id: factId,
            fact,
            category: options.category || 'general',
            source: options.source || 'conversation',
            confidence: options.confidence || 0.8,
            times_used: 0,
            last_used: null,
            created_at: new Date().toISOString()
        };

        this.factsCache.set(factId, factData);

        if (this.db) {
            this.db.prepare(`
                INSERT OR REPLACE INTO facts (id, fact, category, source, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            `).run(factId, fact, factData.category, factData.source, factData.confidence, factData.created_at);
        } else if (this.useJsonFallback) {
            await this.saveJsonFacts();
        }

        // Generate embedding for semantic search
        const embedding = await this.generateEmbedding(fact);
        await this.storeEmbedding(factId, embedding, 'fact');

        this.logger.info(`Stored fact: ${fact.substring(0, 50)}...`);
        return factId;
    }

    /**
     * Extract facts from conversation
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
        return Array.from(this.factsCache.values())
            .filter(f => f.category === category)
            .sort((a, b) => b.confidence - a.confidence);
    }

    /**
     * Get all facts formatted
     */
    getFactsSummary() {
        const factsByCategory = {};

        for (const fact of this.factsCache.values()) {
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
    // SEMANTIC SEARCH
    // ==========================================

    /**
     * Semantic search using embeddings
     */
    async semanticSearch(query, options = {}) {
        const type = options.type || 'all'; // 'documents', 'facts', 'all'
        const limit = options.limit || this.config.maxDocsPerQuery;

        // Generate query embedding
        const queryEmbedding = await this.generateEmbedding(query);

        if (!queryEmbedding) {
            // Fallback to text search
            return this.textSearch(query, limit);
        }

        const results = [];

        // Search through embeddings
        for (const [targetId, embedding] of this.embeddingsCache) {
            const similarity = this.cosineSimilarity(queryEmbedding, embedding);

            if (similarity >= this.config.similarityThreshold) {
                // Determine if it's a document, chunk, or fact
                let item = null;
                let itemType = 'unknown';

                if (targetId.includes('_chunk_')) {
                    // It's a chunk - get parent document
                    const docId = targetId.split('_chunk_')[0];
                    item = this.documentsCache.get(docId);
                    itemType = 'chunk';
                } else if (this.documentsCache.has(targetId)) {
                    item = this.documentsCache.get(targetId);
                    itemType = 'document';
                } else if (this.factsCache.has(targetId)) {
                    item = this.factsCache.get(targetId);
                    itemType = 'fact';
                }

                if (item && (type === 'all' || type === itemType + 's')) {
                    results.push({
                        id: targetId,
                        type: itemType,
                        item,
                        similarity
                    });
                }
            }
        }

        // Sort by similarity and limit
        return results
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, limit);
    }

    /**
     * Fallback text search
     */
    textSearch(query, limit = 5) {
        const queryWords = this.tokenize(query.toLowerCase());
        const scores = new Map();

        // Score documents
        for (const [docId, doc] of this.documentsCache) {
            let score = 0;
            const contentLower = (doc.content || '').toLowerCase();

            for (const word of queryWords) {
                if (contentLower.includes(word)) {
                    score += 1;
                }
            }

            if (score > 0) {
                scores.set(docId, { type: 'document', item: doc, score });
            }
        }

        // Score facts
        for (const [factId, fact] of this.factsCache) {
            let score = 0;
            const factLower = fact.fact.toLowerCase();

            for (const word of queryWords) {
                if (factLower.includes(word)) {
                    score += 1;
                }
            }

            score *= fact.confidence;

            if (score > 0) {
                scores.set(factId, { type: 'fact', item: fact, score });
            }
        }

        return Array.from(scores.entries())
            .map(([id, data]) => ({ id, ...data, similarity: data.score / queryWords.length }))
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, limit);
    }

    /**
     * Search documents (legacy compatibility)
     */
    search(query, limit = 5) {
        return this.textSearch(query, limit)
            .filter(r => r.type === 'document')
            .map(r => ({ document: r.item, score: r.similarity }));
    }

    /**
     * Search facts (legacy compatibility)
     */
    searchFacts(query, limit = 10) {
        return this.textSearch(query, limit)
            .filter(r => r.type === 'fact')
            .map(r => r.item);
    }

    /**
     * Get relevant context for a query (main RAG method)
     */
    async getRelevantContext(query, options = {}) {
        const maxTokens = options.maxTokens || this.config.maxContextTokens;
        let context = '';
        let tokenEstimate = 0;

        // Try semantic search first
        const results = await this.semanticSearch(query, {
            limit: this.config.maxDocsPerQuery + this.config.maxFactsPerQuery
        });

        // Separate documents and facts
        const docs = results.filter(r => r.type === 'document' || r.type === 'chunk').slice(0, this.config.maxDocsPerQuery);
        const facts = results.filter(r => r.type === 'fact').slice(0, this.config.maxFactsPerQuery);

        // Add relevant documents
        if (docs.length > 0) {
            context += '=== RELEVANT DOCUMENTS ===\n\n';
            for (const { item, similarity } of docs) {
                // Use summary if available, otherwise excerpt
                const text = item.summary || item.content?.substring(0, 500) || '';
                const title = item.filename || item.title || 'Document';

                const section = `[${title}] (relevance: ${(similarity * 100).toFixed(0)}%)\n${text}\n\n`;

                // Estimate tokens (rough: 4 chars = 1 token)
                if (tokenEstimate + section.length / 4 < maxTokens) {
                    context += section;
                    tokenEstimate += section.length / 4;
                }
            }
        }

        // Add relevant facts
        if (facts.length > 0) {
            context += '=== KNOWN FACTS ===\n';
            for (const { item } of facts) {
                const factLine = `- ${item.fact}\n`;
                if (tokenEstimate + factLine.length / 4 < maxTokens) {
                    context += factLine;
                    tokenEstimate += factLine.length / 4;
                }
            }
            context += '\n';
        }

        // Add user profile if room
        if (Object.keys(this.userProfile).length > 0 && tokenEstimate < maxTokens * 0.8) {
            context += '=== USER PROFILE ===\n';
            context += JSON.stringify(this.userProfile, null, 2);
            context += '\n\n';
        }

        return context;
    }

    // ==========================================
    // USER PROFILE
    // ==========================================

    async updateProfile(updates) {
        this.userProfile = {
            ...this.userProfile,
            ...updates,
            lastUpdated: new Date().toISOString()
        };
        await this.saveProfile();
    }

    async setProfileField(field, value) {
        this.userProfile[field] = value;
        this.userProfile.lastUpdated = new Date().toISOString();
        await this.saveProfile();
    }

    getProfileField(field) {
        return this.userProfile[field];
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

    // ==========================================
    // CONVERSATION MEMORY
    // ==========================================

    /**
     * Store conversation summary for long-term memory
     */
    async storeConversationSummary(userId, messages) {
        if (!this.callOllama || messages.length < 4) return;

        try {
            const conversationText = messages.map(m => `${m.role}: ${m.content}`).join('\n');

            const prompt = `Summarize this conversation in 2-3 sentences, noting key topics discussed and any important information revealed:\n\n${conversationText}`;

            const summary = await this.callOllama('researcher', prompt);

            if (this.db) {
                const convId = this.generateId(`conv_${userId}_${Date.now()}`);
                this.db.prepare(`
                    INSERT INTO conversations (id, user_id, summary, message_count)
                    VALUES (?, ?, ?, ?)
                `).run(convId, userId, summary, messages.length);
            }

            // Also extract facts
            await this.extractFactsFromConversation(messages);

        } catch (error) {
            this.logger.warn('Failed to store conversation summary:', error.message);
        }
    }

    // ==========================================
    // HELPERS
    // ==========================================

    generateId(input) {
        return crypto.createHash('md5').update(input + Date.now()).digest('hex').substring(0, 12);
    }

    tokenize(text) {
        return text
            .toLowerCase()
            .replace(/[^\w\s]/g, ' ')
            .split(/\s+/)
            .filter(word => word.length > 2);
    }

    chunkText(text, chunkSize = null, overlap = null) {
        chunkSize = chunkSize || this.config.chunkSize;
        overlap = overlap || this.config.chunkOverlap;

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
    // PERSISTENCE (JSON Fallback)
    // ==========================================

    async saveJsonDocuments() {
        if (!this.useJsonFallback) return;
        const data = Object.fromEntries(this.documentsCache);
        await fs.writeFile(this.jsonFiles.documents, JSON.stringify(data, null, 2));
    }

    async saveJsonFacts() {
        if (!this.useJsonFallback) return;
        const data = Object.fromEntries(this.factsCache);
        await fs.writeFile(this.jsonFiles.facts, JSON.stringify(data, null, 2));
    }

    async saveJsonEmbeddings() {
        if (!this.useJsonFallback) return;
        const data = Object.fromEntries(this.embeddingsCache);
        await fs.writeFile(this.jsonFiles.embeddings, JSON.stringify(data, null, 2));
    }

    /**
     * Get statistics
     */
    getStats() {
        return {
            documents: this.documentsCache.size,
            facts: this.factsCache.size,
            embeddingsLoaded: this.embeddingsCache.size,
            profileFields: Object.keys(this.userProfile).length,
            usingDatabase: !!this.db,
            embeddingModel: this.embeddingModel
        };
    }

    /**
     * Close database connection
     */
    close() {
        if (this.db) {
            this.db.close();
            this.db = null;
        }
    }
}

module.exports = PersonalKnowledgeBase;
