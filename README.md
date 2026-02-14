# Wang Yu's AI Portfolio

A minimalist personal portfolio featuring an integrated AI assistant, local RAG (Retrieval-Augmented Generation), and animations.

## ✨ Features
- **Minimalist Terminal UI**: A sleek, dark-mode terminal interface for interaction.
- **Local AI Assistant**: Powered by `Transformers.js` (`Qwen2.5-0.5B-Instruct`) running entirely in your browser.
- **Local RAG**: Context-aware answers derived from a structured knowledge base (`portfolio.json`) using vector search.
- **Styled Loader**: High-performance morphing animation inspired by minimalist aesthetics.
- **Privacy First**: No data leaves your machine; all inference is performed locally.

## 🛠 Tech Stack
- **Frontend**: HTML5, Vanilla CSS, Javascript.
- **Static Site Generator**: [Eleventy (11ty)](https://www.11ty.dev/).
- **AI/ML**: [@huggingface/transformers](https://huggingface.co/docs/transformers.js).
- **Search**: Vector-based semantic retrieval.

## 🚀 Getting Started

### Prerequisites
- [Node.js](https://nodejs.org/) (v18+ recommended)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/oneuuuu/oneuuuu.github.io.git
   ```
2. Install dependencies:
   ```bash
   npm install
   ```

### Development
Start the local development server:
```bash
npm start
```
The site will be available at `http://localhost:8080`.

### Generating Embeddings
If you update `src/data/portfolio.json`, you must regenerate the knowledge base:
```bash
npm run embed
```

## 📄 License
ISC License. See `LICENSE` for details.
