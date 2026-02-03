import "dotenv/config";

import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { QdrantVectorStore } from "@langchain/qdrant";

async function run() {
  try {
    if (!process.env.GOOGLE_API_KEY) {
      throw new Error("GOOGLE_API_KEY is missing");
    }

    if (!process.env.QDRANT_URL || !process.env.QDRANT_API_KEY) {
      throw new Error("QDRANT credentials are missing");
    }

    // 1️⃣ Load PDF
    const loader = new PDFLoader("./PORTFOLIORAGDOC.pdf");
    const docs = await loader.load();

    // 2️⃣ Split into chunks
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 450,
      chunkOverlap: 70,
    });

    const chunkedDocs = await splitter.splitDocuments(docs);
    console.log(`✅ Step 1: ${chunkedDocs.length} chunks ready.`);

    // 3️⃣ Embeddings (used internally by QdrantVectorStore)
    const embeddings = new GoogleGenerativeAIEmbeddings({
      model: "text-embedding-004",
    });

    // 4️⃣ Connect to existing Qdrant collection
    const vectorStore = await QdrantVectorStore.fromExistingCollection(
      embeddings,
      {
        url: process.env.QDRANT_URL,
        apiKey: process.env.QDRANT_API_KEY,
        collectionName: "langchainjs-testing", // ✅ MUST MATCH QDRANT
      }
    );

    // 5️⃣ Store documents (embeddings happen here)
    await vectorStore.addDocuments(chunkedDocs);

    console.log("✅ Step 2: Documents embedded and stored in Qdrant.");
  } catch (error) {
    console.error("❌ SCRIPT CRASHED:", error.message);
  }
}

run();
