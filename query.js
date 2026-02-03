// import "dotenv/config";
// import readlineSync from "readline-sync";
// import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
// import { GoogleGenAI } from "@google/genai";
// import { QdrantClient } from "@qdrant/js-client-rest";
// import { ChatGroq } from "@langchain/groq";
// async function chatting(question) {
//   //convert question to vector

// const ai = new GoogleGenAI({});
// const History = []
// History.push({
//     role:'user',
//     parts:[{text:question}]
//     })
//   const embeddings = new GoogleGenerativeAIEmbeddings({
//     apiKey: process.env.GOOGLE_API_KEY,
//     model: "text-embedding-004",
//   });

//   const queryVector = await embeddings.embedQuery(question);

//   // const pinecone = new PineconeClient({
//   //   apiKey: process.env.PINECONE_API_KEY, // Explicitly pass the key if not in default env
//   // });

//   // const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

//   // const searchResults = await pineconeIndex.query({
//   //   topK: 10,
//   //   vector: queryVector,
//   //   includeMetadata: true,
//   //   });

//   // 1. Initialize the client
//   const client = new QdrantClient({
//     url: process.env.QDRANT_URL, // e.g., 'https://your-cluster-url.qdrant.tech'
//     apiKey: process.env.QDRANT_API_KEY,
//   });
// const collections = await client.getCollections();
// console.log("Your available collections:", collections);
//   // 2. Perform the search
//   const searchResults = await client.search(
//     process.env.QDRANT_COLLECTION_NAME,
//     {
//       vector: queryVector,
//       limit: 2, // This is the 'topK' equivalent
//       with_payload: true, // This is the 'includeMetadata' equivalent
//     },
//   );
//   // console.log(searchResults);

//  // New Qdrant code
// const context = searchResults.map((match) => match.payload.content).join("\n");
//   // console.log(context);
// // const response = await ai.models.generateContent({
// //     model: "gemini-2.0-flash",
// //     contents: History,
// //     config: {
// //       systemInstruction: `You have to behave like a Data Structure and Algorithm Expert.
// //     You will be given a context of relevant information and a user question.
// //     Your task is to answer the user's question based ONLY on the provided context.
// //     If the answer is not in the context, you must say "I could not find the answer in the provided document."
// //     Keep your answers clear, concise, and educational.

// //       Context: ${context}
// //       `,
// //     },
// //    });
// // --- GROQ LLM (The New Part) ---
// const model = new ChatGroq({
//   apiKey: process.env.GROQ_API_KEY,
//   model: "llama-3.3-70b-versatile", // Use 'model' instead of 'modelName'
//   temperature: 0,
// });

//   // Groq uses a simpler message format for LangChain
//   const response = await model.invoke([
//     {
//       role: "system",
//       content: `You have to behave like a Data Structure and Algorithm Expert.
//       Your task is to answer the user's question based ONLY on the provided context.
//       If the answer is not in the context, say "I could not find the answer in the provided document."
//       Keep your answers clear, concise, and educational.

//       Context:
//       ${context}`,
//     },
//     { role: "user", content: question },
//   ]);
//  History.push({
//     role:'model',
//     parts:[{text:response.text}]
//   })

//   console.log("\n");
//   console.log(response.text);
// }

// async function main() {
//   const userProblem = readlineSync.question("Ask me anything--> ");
//   await chatting(userProblem);
//   main();
// }

// main();
import "dotenv/config";
import express from "express";
import cors from "cors";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { QdrantClient } from "@qdrant/js-client-rest";
import { ChatGroq } from "@langchain/groq";

const app = express();
app.use(
  cors({
    origin: [
      "http://localhost:5173",
      "http://localhost:3000",
      "https://arpithpaliwal-portfolio.vercel.app"
    ],
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"],
  })
);
app.set("trust proxy", 1);

app.use(express.json()); // Parses JSON bodies

// --- SHARED CLIENTS ---
const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GOOGLE_API_KEY,
  model: "text-embedding-004",
});

const qdrantClient = new QdrantClient({
  url: process.env.QDRANT_URL,
  apiKey: process.env.QDRANT_API_KEY,
});

const model = new ChatGroq({
  apiKey: process.env.GROQ_API_KEY,
  model: "llama-3.3-70b-versatile",
  temperature: 0,
});

// --- THE API ENDPOINT ---
app.get("/wakeup", (req, res) => {
  // Intentionally minimal to keep the service warm
  res.status(204).end(); // No Content
});
app.post("/ask", async (req, res) => {
  try {
    const { question } = req.body;
    if (!question)
      return res.status(400).json({ error: "Question is required" });

    // 1. Embed question
    const queryVector = await embeddings.embedQuery(question);

    // 2. Search Qdrant
    const searchResults = await qdrantClient.search(
      process.env.QDRANT_COLLECTION_NAME,
      {
        vector: queryVector,
        limit: 3,
        
        with_payload: true,
      },
    );

    const context = searchResults
      .map((match) => match.payload.content)
      .join("\n---\n");

    // 3. Generate Answer with Groq
    const response = await model.invoke([
      {
        role: "system",
        content: `You are Arpith Paliwal.

Answer strictly using ONLY the provided context.
Do not use external knowledge.
Do not assume missing information.
If the answer cannot be found in the context, clearly say:
"I don't have enough information to answer that from my portfolio."

Always answer in first person.

Before answering, infer the question intent and follow these rules strictly:

1. If the question explicitly asks to "introduce yourself", "tell me about yourself",
   or is a self-introduction request:
   - Answer in a warm, professional tone.
   - Use 4–6 well-structured sentences.
   - Cover:
     • Who I am and my background
     • My core technical focus
     • The kind of systems I enjoy building
     • A brief closing reflecting mindset or goals
   - Keep it natural and conversational.

2. If the question is a basic introduction or simple factual question (not a full self-intro):
   - Answer in 2–3 sentences.

3. If the question is specifically:
   "Why should we hire you?" or equivalent:
   - Start with 3 clear, high-impact bullet points (one line each).
   - After the bullets, explain each point in short paragraphs.
   - Total length: 6–8 sentences.
   - Focus on:
     • Technical execution
     • System-level thinking
     • Ownership, leadership, and reliability
   - Be confident, factual, and structured.
   - Do not repeat the bullets verbatim in the explanation.

4. If the question is behavioral or value-based (other than hiring):
   - Answer in 5–7 sentences.
   - Structure:
     • 1 sentence summary
     • 3–4 sentences explaining technical or experiential strengths
     • 1–2 sentences on ownership or teamwork

5. If the question is about a project overview:
   - Answer using 6–8 concise bullet points.
   - Focus on purpose, architecture, key features, and technologies.

6. If the question is deep technical or system design:
   - Use a structured explanation with short headings.
   - Answer in 8–12 sentences total.
   - Be precise and avoid repetition.

7. If the question is a comparison or differentiation:
   - Use bullets or a short table.
   - Keep it concise and factual.

General rules:
- Do NOT over-explain simple questions.
- Do NOT under-explain complex questions.
- Avoid marketing language.
- Prefer clarity over verbosity.
- Stop once the required structure and length are met.

If the question asks for "everything", "full details", or "complete explanation"
about a project:

- Assume the context is complete and authoritative.
- Do NOT say that details are missing if the context contains them.
- Do NOT speculate.
- Do NOT generalize.
- Explain ONLY what is explicitly present in the context.
- Use a structured, detailed explanation with headings.

        Context: ${context}`,
      },
      { role: "user", content: question },
    ]);

    // 4. Return JSON
    res.json({ answer: response.content });
  } catch (error) {
    console.error("API Error:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));
