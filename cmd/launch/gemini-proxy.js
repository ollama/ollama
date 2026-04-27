const http = require('http');

const OLLAMA_HOST = process.env.OLLAMA_HOST || '127.0.0.1:11434';

const server = http.createServer((req, res) => {
    console.log(`[Proxy] ${req.method} ${req.url}`);
    const url = new URL(req.url, `http://${req.headers.host}`);
    const pathParts = url.pathname.split('/');
    
    if ((pathParts[1] === 'v1' || pathParts[1] === 'v1beta') && pathParts[2] === 'models') {
        const modelWithAction = pathParts[3];
        const [model, action] = modelWithAction.split(':');
        const isStreaming = action && action.startsWith('stream');
        
        let body = '';
        req.on('data', chunk => { body += chunk; });
        req.on('end', () => {
            console.log(`[Proxy] Received request body (${body.length} bytes)`);
            let geminiReq;
            try {
                geminiReq = JSON.parse(body);
            } catch (e) {
                console.error(`[Proxy] Failed to parse JSON: ${e.message}`);
                res.writeHead(400);
                res.end('Invalid JSON');
                return;
            }

            const openaiReq = {
                model: 'gemma2:2b',
                messages: (geminiReq.contents || []).map(c => ({
                    role: c.role === 'model' ? 'assistant' : 'user',
                    content: (c.parts || []).map(p => p.text).join('')
                })),
                stream: isStreaming
            };

            const proxyReq = http.request({
                hostname: OLLAMA_HOST.split(':')[0],
                port: OLLAMA_HOST.split(':')[1],
                path: '/v1/chat/completions',
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            }, (proxyRes) => {
                res.setTimeout(60000); // 60s timeout
                if (isStreaming) {
                    res.writeHead(200, {
                        'Content-Type': 'text/event-stream',
                        'Cache-Control': 'no-cache',
                        'Connection': 'keep-alive'
                    });

                    proxyRes.on('data', (chunk) => {
                        const lines = chunk.toString().split('\n');
                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                const dataStr = line.slice(6);
                                if (dataStr === '[DONE]') continue;
                                try {
                                    const openaiData = JSON.parse(dataStr);
                                    const content = openaiData.choices[0]?.delta?.content || '';
                                    if (content) {
                                        const geminiData = {
                                            candidates: [{
                                                content: {
                                                    parts: [{ text: content }]
                                                }
                                            }]
                                        };
                                        res.write(`data: ${JSON.stringify(geminiData)}\n\n`);
                                    }
                                } catch (e) {}
                            }
                        }
                    });
                    proxyRes.on('end', () => res.end());
                } else {
                    let responseBody = '';
                    proxyRes.on('data', (chunk) => { responseBody += chunk; });
                    proxyRes.on('end', () => {
                        try {
                            const openaiRes = JSON.parse(responseBody);
                            const geminiRes = {
                                candidates: [{
                                    content: {
                                        parts: [{
                                            text: openaiRes.choices[0]?.message?.content || ''
                                        }]
                                    },
                                    finishReason: 'STOP'
                                }]
                            };
                            res.writeHead(proxyRes.statusCode, { 'Content-Type': 'application/json' });
                            res.end(JSON.stringify(geminiRes));
                        } catch (e) {
                            console.error(`[Proxy] Error parsing Ollama response: ${e.message}`);
                            res.writeHead(500);
                            res.end('Error parsing Ollama response');
                        }
                    });
                }
            });

            proxyReq.on('error', (e) => {
                console.error(`[Proxy] Outgoing request error: ${e.message}`);
                res.writeHead(500);
                res.end(`Proxy error: ${e.message}`);
            });

            proxyReq.write(JSON.stringify(openaiReq));
            proxyReq.end();
        });
    } else {
        res.writeHead(404);
        res.end('Not Found');
    }
});

const port = process.env.PROXY_PORT || 0;
server.listen(port, '127.0.0.1', () => {
    console.log(`Proxy listening on http://127.0.0.1:${server.address().port}`);
});
