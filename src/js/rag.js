// rag.js - Proxy for worker.js
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });

let onProgressUpdate = null;
let askResolve = null;
let askOnStream = null;

export function setOnProgressUpdate(callback) {
    onProgressUpdate = callback;
}

worker.onmessage = (e) => {
    const { type, progress, text, error } = e.data;

    if (type === 'progress' && onProgressUpdate) {
        onProgressUpdate(progress, 'HYBRID ENGINE');
    } else if (type === 'init-complete') {
        if (initResolve) initResolve();
    } else if (type === 'stream' && askOnStream) {
        askOnStream(text);
    } else if (type === 'done') {
        if (askResolve) askResolve(text);
    } else if (type === 'error') {
        console.error('Worker Error:', error);
    }
};

let initResolve = null;
export function preloadModels() {
    return new Promise((resolve) => {
        initResolve = resolve;
        worker.postMessage({ type: 'init' });
    });
}

export function ask(query, onStream) {
    return new Promise((resolve) => {
        askResolve = resolve;
        askOnStream = onStream;
        worker.postMessage({ type: 'ask', query });
    });
}
