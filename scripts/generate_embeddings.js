
import { pipeline } from '@huggingface/transformers';
import fs from 'fs';
import path from 'path';

const DATA_DIR = './src/data';
const PORTFOLIO_FILE = path.join(DATA_DIR, 'portfolio.json');
const OUTPUT_FILE = path.join(DATA_DIR, 'embeddings.json');

async function generateEmbeddings() {
    console.log('Starting embedding generation...');

    const portfolio = JSON.parse(fs.readFileSync(PORTFOLIO_FILE, 'utf8'));

    const chunks = [];

    // Add "About"
    chunks.push({
        id: 'about',
        content: portfolio.about,
        metadata: { section: 'about' }
    });

    // Add Experience
    portfolio.experience.forEach((ext, index) => {
        chunks.push({
            id: `experience-${index}`,
            content: `${ext.role} at ${ext.company} (${ext.period}): ${ext.description}`,
            metadata: { section: 'experience', ...ext }
        });
    });

    // Add Projects
    portfolio.projects.forEach((proj, index) => {
        chunks.push({
            id: `project-${index}`,
            content: `Project ${proj.name}: ${proj.description}. Technologies used: ${proj.technologies.join(', ')}`,
            metadata: { section: 'projects', ...proj }
        });
    });

    // Add Skills
    chunks.push({
        id: 'skills',
        content: `My skills include: ${portfolio.skills.join(', ')}`,
        metadata: { section: 'skills' }
    });

    // Add Education
    portfolio.education.forEach((edu, index) => {
        const content = edu.degree ? `${edu.degree} at ${edu.school} (Graduated ${edu.year})` : `Certification: ${edu.certification}`;
        chunks.push({
            id: `education-${index}`,
            content: content,
            metadata: { section: 'education', ...edu }
        });
    });

    console.log(`Prepared ${chunks.length} chunks. Loading embedding model...`);

    const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

    const embeddings = [];
    for (const chunk of chunks) {
        console.log(`Generating embedding for: ${chunk.id}`);
        const output = await embedder(chunk.content, { pooling: 'mean', normalize: true });
        embeddings.push({
            ...chunk,
            embedding: Array.from(output.data)
        });
    }

    fs.writeFileSync(OUTPUT_FILE, JSON.stringify(embeddings, null, 2));
    console.log(`Embeddings saved to ${OUTPUT_FILE}`);
}

generateEmbeddings().catch(console.error);
