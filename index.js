import "dotenv/config";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { Pinecone as PineconeClient } from "@pinecone-database/pinecone";
import { QdrantVectorStore } from "@langchain/qdrant";
async function run() {
  try {
    // 1. Load and Split
    const loader = new PDFLoader("./PORTFOLIORAGDOC.pdf");
    const docs = await loader.load();
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 400,
      chunkOverlap: 50,
    });
    const chunkedDocs = await splitter.splitDocuments(docs);
    console.log(`✅ Step 1: ${chunkedDocs.length} chunks ready.`);

    // 2. Embed
    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GOOGLE_API_KEY,
      model: "text-embedding-004",
    });
    const vectors = await embeddings.embedDocuments(
      chunkedDocs.map((d) => d.pageContent),
    );
    console.log(`✅ Step 2: ${vectors.length} vectors generated.`);
    const vectorStore = await QdrantVectorStore.fromExistingCollection(
  embeddings,
  {
    url: process.env.QDRANT_URL,
    apiKey: process.env.QDRANT_API_KEY,
    collectionName: "langchainjs-testing",
  }
);

    await vectorStore.addDocuments(chunkedDocs);
    // const client = new QdrantClient({
    //     url: 'https://3db886d2-27bf-43ba-be18-21cdaf81bda2.us-east4-0.gcp.cloud.qdrant.io:6333',
    //     apiKey: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Fur8Z1opdFLS0z2cNH-G5WKayMFWSCK96IDC4ILmoXk',
    // });
  } catch (error) {
    console.error("❌ SCRIPT CRASHED:", error.message);
  }
}

run();
