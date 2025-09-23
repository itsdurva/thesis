"""fedrag: A Flower Federated RAG app."""

from flwr.client import ClientApp
from flwr.common import ConfigRecord, Context, Message, RecordDict

from fedrag.retriever import Retriever


# Flower ClientApp
app = ClientApp()
print("init init")
    

@app.query()
def query(msg: Message, context: Context):

    node_id = context.node_id

    # Extract question
    questions = msg.content["config"]["question"]
    q_ids = msg.content["config"]["question_id"]
    qa_dataset_name = str(msg.content["config"]["question_dataset"])
    # Extract corpus name
    corpus_name = str(msg.content["config"]["corpus_name"])

    # Initialize retrieval system
    retriever = Retriever()
    # Use the knn value for retrieving the closest-k documents to the query
    knn = int(msg.content["config"]["knn"])
    # change here
    retrieved_docs = retriever.query_faiss_index(corpus_name, qa_dataset_name, questions, knn)

    all_documents = []
    all_scores = []
    # Create lists with the computed scores and documents
    for q, question_id in enumerate(questions):
        scores = [doc["score"] for doc_id, doc in retrieved_docs[q].items()]
        all_scores.extend(scores)
        documents = [doc["content"] for doc_id, doc in retrieved_docs[q].items()]
        all_documents.extend(documents)
        print(
            "ClientApp: {} - Question: {} - Retrieved: {} documents.".format(
                node_id, str(q_ids+q), len(documents)
            )
        )

    # Create reply record with retrieved documents.
        docs_n_scores = ConfigRecord(
            {
                "documents": all_documents,
                "scores": all_scores,
            }
        )
    reply_record = RecordDict({"docs_n_scores": docs_n_scores})

    # Return message
    return Message(reply_record, reply_to=msg)
