import { pipeline, TextStreamer } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.2';

let embeddingsData = null;
let embedder = null;
let generator = null;

async function loadData() {
    if (!embeddingsData) {
        const response = await fetch('/data/embeddings.json');
        embeddingsData = await response.json();
    }
    return embeddingsData;
}

async function getEmbedder() {
    if (!embedder) {
        embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    }
    return embedder;
}

async function getEngine() {
    if (!generator) {
        const selectedModel = 'onnx-community/Qwen2.5-0.5B-Instruct';
        generator = await pipeline('text-generation', selectedModel, {
            device: 'webgpu',
            dtype: 'q4'
        });
    }
    return generator;
}

function cosineSimilarity(v1, v2) {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < v1.length; i++) {
        dotProduct += v1[i] * v2[i];
        normA += v1[i] * v1[i];
        normB += v2[i] * v2[i];
    }
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function search(query, topK = 10) {
    const data = await loadData();
    const currentEmbedder = await getEmbedder();
    const output = await currentEmbedder(query, { pooling: 'mean', normalize: true });
    const queryEmbedding = Array.from(output.data);

    const scores = data.map(chunk => ({
        ...chunk,
        score: cosineSimilarity(queryEmbedding, chunk.embedding)
    }));

    return scores.sort((a, b) => b.score - a.score).slice(0, topK);
}

self.onmessage = async (e) => {
    const { type, query } = e.data;

    if (type === 'init') {
        try {
            await Promise.all([getEmbedder(), getEngine(), loadData()]);
            self.postMessage({ type: 'init-complete' });
        } catch (err) {
            self.postMessage({ type: 'error', error: err.message });
        }
    } else if (type === 'ask') {
        try {
            const results = await search(query);
            const context = results.map(r => r.content).join('\n\n');
            const chatGenerator = await getEngine();

            const messages = [
                {
                    role: "system",
                    content: `You are a professional AI assistant for Wang Yu's portfolio. 
[DATA MAP]
- "he", "him", "his" -> refers to Wang Yu.
- "you", "your" -> refers to Wang Yu (the person the user is asking about).

[RULES]
1. Answer ONLY using the facts in the [CONTEXT] below.
2. If info is missing, say "Data not found in records."
3. LISTS: When listing companies or skills, provide a UNIQUE, comma-separated list.
4. NO INFERENCE: Do not guess or use outside knowledge.

[CONTEXT]
${context}`
                },
                { role: "user", content: query }
            ];

            const prompt = chatGenerator.tokenizer.apply_chat_template(messages, { tokenize: false, add_generation_prompt: true });

            let fullAnswer = "";
            const streamer = new TextStreamer(chatGenerator.tokenizer, {
                skip_prompt: true,
                callback_function: (text) => {
                    fullAnswer += text;
                    self.postMessage({ type: 'stream', text: fullAnswer });
                }
            });

            await chatGenerator(prompt, {
                max_new_tokens: 256,
                streamer,
                do_sample: false,
                temperature: 0.0
            });

            self.postMessage({ type: 'done', text: fullAnswer });
        } catch (err) {
            self.postMessage({ type: 'error', error: err.message });
        }
    }
};
