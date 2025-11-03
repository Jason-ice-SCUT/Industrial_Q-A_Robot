import os_mirror.os_mirror # 个人构建的hf镜像环境
import chunk_traditional as chunk_t
import chromadb
from FlagEmbedding import FlagAutoModel


EMBEDDING_MODEL = FlagAutoModel.from_finetuned('BAAI/bge-base-en-v1.5',
                                      query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                                      use_fp16=True)
#LLM_MODEL = 

chromadb_client = chromadb.PersistentClient("./chroma.db")
chromadb_collection = chromadb_client.get_or_create_collection("IndustrialQA_DB")

def create_db() ->None:
    for idx, c in enumerate(chunk_t.chunk_text()):
        print(f"Process: {c}")
        embedding: list[float] = embed_text(c, for_query=True)
        chromadb_collection.upsert(
            ids=str(idx),
            documents=c,
            embeddings=embedding
        )

# Define the function to generate embeddings for a given text 
def embed_text(text: str, for_query: bool) -> list[float]:
    model = EMBEDDING_MODEL
    if for_query:
        embeddings = model.encode_queries([text])
        embeddings_list = embeddings.tolist()

        assert embeddings_list
        assert embeddings_list[0]
        return embeddings_list[0]
        
    else:
        embeddings = model.encode_corpus([text])
        embeddings_list = embeddings.tolist()

        assert embeddings_list
        assert embeddings_list[0]
        return embeddings_list[0]
        
  
def query_db(query: str) -> list[dict]:
    embedding = embed_text(query, for_query=True)
    results = chromadb_collection.query(
        query_embeddings= embedding,
        n_results=5
    )
    assert results['documents']
    return results['documents'][0]

if __name__ == "__main__":
   query = "电动平衡车的安全要求是什么？"
   #create_db()
   chunks: list[str] = query_db(query)

   prompt = "Please answer the question based on the following context:\n"
   prompt += f"Query: {query}\n"
   prompt += "Context:\n"   
   for c in chunks:
       prompt += c + "\n"
       prompt += "-----\n"

   print(prompt)
   
