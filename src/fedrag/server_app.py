"""fedrag: A Flower Federated RAG app."""
import os, glob, time, hashlib, json, numpy as np
from itertools import cycle
from collections import defaultdict
from time import sleep
from datetime import datetime

from fedrag.llm_querier import LLMQuerier
from fedrag.mirage_qa import MirageQA
from fedrag.task import index_exists
from sklearn.metrics import accuracy_score

from flwr.common import ConfigRecord, Context, Message, MessageType, RecordDict
from flwr.server import Grid, ServerApp
from sentence_transformers import CrossEncoder

def node_online_loop(grid: Grid) -> list[int]:
    node_ids = []
    while not node_ids:
        # Get IDs of nodes available
        node_ids = grid.get_node_ids()
        # Wait if no node is available
        sleep(1)
    return node_ids


def get_hash(doc):
    # Create and return an SHA-256 hash for the given document
    return hashlib.sha256(doc.encode())


def merge_documents(question, documents, scores, knn, rerank_model, k_rrf=0, reverse_sort=False) -> list[str]:
    RRF_dict = defaultdict(dict)
    RERANK_dict = defaultdict(dict)
    sorted_scores = np.array(scores).argsort()
    if reverse_sort:  # from larger to smaller scores
        sorted_scores = sorted_scores[::-1]
    sorted_documents = [documents[i] for i in sorted_scores]

    if k_rrf == 0:
        # If k_rff is not set then simply return the
        # sorted documents based on their retrieval score
        return sorted_documents[:knn]
    
    # adding code for reranking model
    if k_rrf == -1:
        ce = CrossEncoder(rerank_model)
        pairs = [(question, doc) for doc in sorted_documents]
        scores = ce.predict(pairs)  # Single batch call
        for doc, score in zip(sorted_documents, scores):
            RERANK_dict[get_hash(doc)] = {"score": score, "doc": doc}
            # the more similar, the better
            RERANK_docs = sorted(RERANK_dict.values(), key=lambda x: x["score"], reverse=True)
            docs = [rrf_res["doc"] for rrf_res in RERANK_docs][:knn]  
        return docs

    for doc_idx, doc in enumerate(sorted_documents):
        # Given that some returned results/documents could be extremely
        # large we cannot use the original document as a dictionary key.
        # Therefore, we first hash the returned string/document to a
        # representative hash code, and we use that code as a key for
        # the final RRF dictionary. We follow this approach, because a
        # document could  have been retrieved twice by multiple clients
        # but with different scores and depending on these scores we need
        # to maintain its ranking
        doc_hash = get_hash(doc)
        RRF_dict[doc_hash]["rank"] = 1 / (k_rrf + doc_idx + 1)
        RRF_dict[doc_hash]["doc"] = doc

    RRF_docs = sorted(RRF_dict.values(), key=lambda x: x["rank"], reverse=True)
    docs = [rrf_res["doc"] for rrf_res in RRF_docs][
        :knn
    ]  # select the final top-k / k-nn
    return docs


def submit_questions(
    grid: Grid,
    dataset_name: str,
    question: list,
    q_ids: str,
    knn: int,
    node_ids: list,
    corpus_names_iter: iter,
):

    messages = []
    # Send the same Message to each connected node (which run `ClientApp` instances)
    for node_idx, node_id in enumerate(node_ids):
        # The payload of a Message is of type RecordDict
        # https://flower.ai/docs/framework/ref-api/flwr.common.RecordDict.html
        # which can carry different types of records. We'll use a ConfigRecord object
        # We need to create a new ConfigRecord() object for every node, otherwise
        # if we just override a single key, e.g., corpus_name, the grid will send
        # the same object to all nodes.
        config_record = ConfigRecord()
        config_record["question_dataset"] = dataset_name
        config_record["question"] = question
        config_record["question_id"] = q_ids
        config_record["knn"] = knn
        # Round-Robin assignment of corpus to individual clients
        # by infinitely looping over the corpus names.
        config_record["corpus_name"] = next(corpus_names_iter)
        task_record = RecordDict({"config": config_record})
        message = Message(
            content=task_record,
            message_type=MessageType.QUERY,  # target `query` method in ClientApp
            dst_node_id=node_id,
            group_id=str(q_ids),
        )
        messages.append(message)

    # Send messages and wait for all results
    replies = grid.send_and_receive(messages)
    print("Received {}/{} results".format(len(replies), len(messages)))

    documents, scores = [], []
    for reply in replies:
        if reply.has_content():
            documents.extend(reply.content["docs_n_scores"]["documents"])
            scores.extend(reply.content["docs_n_scores"]["scores"])

    return documents, scores


app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    # get directory name from file
    directory_path = os.path.dirname(os.path.realpath(__file__))

    # make results directory if it does not exist, create new file
    results_directory = os.path.join(directory_path, "../../results")
    os.makedirs(results_directory, exist_ok=True)
    existing = [int(os.path.splitext(os.path.basename(f))[0]) 
            for f in glob.glob(os.path.join(results_directory, "*.json")) 
            if os.path.splitext(os.path.basename(f))[0].isdigit()]
    filename = max(existing, default=0) + 1
    filepath = os.path.join(results_directory, f"{filename}.json")
    print("storing results @ ", filepath)

    # start storing config
    json_data = dict(context.run_config)
    json_data["starts"] = str(datetime.now())

    # k-reciprocal-rank-fusion is used by the server to merge the results returned by the clients
    k_rrf = int(context.run_config["k-rrf"])

    # k-nearest-neighbors for document retrieval at each client
    knn = int(context.run_config["k-nn"])

    # get client corpus names & create corpus iterator
    corpus_names = context.run_config["clients-corpus-names"].split("|")
    corpus_names = [c.lower() for c in corpus_names]  # make them lower case
    corpus_names_iter = cycle(corpus_names)

    # get qa datasets
    qa_datasets = context.run_config["server-qa-datasets"].split("|")
    qa_datasets = [qad.lower() for qad in qa_datasets]  # make them lower case
    
    # get number of questions from qa datasets to be answered, useful for debugging
    number_of_questions = context.run_config.get("server-qa-num", None)
    
    # questions batch size: questions sent per "message"
    batch_size = int(context.run_config["batch-size"])

    # name of llm generator model
    model_name = context.run_config["server-llm-hfpath"]
    
    # get reranker model (to merge snippets)
    rerank_model = context.run_config.get("server-rerank-hfpath", None)

    # to use gpu or not
    use_gpu = context.run_config.get("server-llm-use-gpu", False)
    use_gpu = True if use_gpu.lower() == "true" else False

    index_exists(corpus_names, "faiss", model_name)

    # clients
    node_ids = node_online_loop(grid)

    # load all questions from benchmark.json
    mirage_file = os.path.join(os.path.dirname(__file__), "../../benchmark.json")
    datasets = {key: MirageQA(key, mirage_file) for key in qa_datasets}
    print("datasets in question", datasets.keys())

    llm_querier = LLMQuerier(model_name, use_gpu)
    expected_answers, predicted_answers, question_times, unanswered_questions = (defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(int),)
    
    for dataset_name in qa_datasets:
        q_idx = 0
        # print("Evaluating Dataset: [{:s}] ".format(dataset_name))
        questions_in_batch = []
        if number_of_questions is None:
            qa_num = len(datasets[dataset_name])
            print("changing qa_num")
        else:
            qa_num = number_of_questions
        for q in datasets[dataset_name]:
            q_idx += 1
            q_id = f"{dataset_name}_{q_idx}"
            # exit question loop if number of questions has been exceeded
            if qa_num and q_idx > qa_num:
                break
            question = q["question"]
            questions_in_batch.append(q)
            q_st = time.time()
            if (q_idx % batch_size == 0) or (q_idx == qa_num):
                print(len(questions_in_batch))
                #print(questions_in_batch)
                questions = [item['question'] for item in questions_in_batch]
                start_qidx = q_idx - batch_size + 1
                docs, scores = submit_questions(
                    grid, dataset_name, questions, start_qidx, knn, node_ids, corpus_names_iter
                )
                for i in range(batch_size):
                    qdocs = docs[(i*knn): (i+1)*knn]
                    qscores = scores[(i*knn): (i+1)*knn]
                    question = questions_in_batch[i]['question']
                    options = questions_in_batch[i]['options']
                    answer = questions_in_batch[i]["answer"]
                    merged_docs = merge_documents(question, qdocs, qscores, knn, rerank_model, k_rrf)
                    prompt, predicted_answer = llm_querier.answer(
                        question, merged_docs, options, dataset_name
                    )

                    # If the model did not predict any value,
                    # then discard the question
                    if predicted_answer is not None:
                        expected_answers[dataset_name].append(answer)
                        predicted_answers[dataset_name].append(predicted_answer)
                        q_et = time.time()
                        q_time = q_et - q_st  # elapsed time in seconds
                        question_times[dataset_name].append(q_time)
                    else:
                        unanswered_questions[dataset_name] += 1
                questions_in_batch = []
                q_ids = []
    """
    print(
        "Below, for each benchmark dataset (QA Dataset), we show: \n"
        "(1) the evaluation results in terms of the total number of Federated RAG queries executed (Total Questions). \n"
        "(2) the total number of queries answered by the LLM when prompted with the retrieved documents from the federation clients (Answered Questions). \n"
        "(3) the overall performance of the Federated RAG pipeline (Accuracy), i.e., expected answer vs. predicted answer by the LLM. \n"
        "(4) the mean wall-clock time (Mean Querying Time) for executing all Federated RAG queries; from the time the server submits the query to "
        "the clients to the time the server receives the final prediction result from the LLM model when prompted with the retrieved documents.\n"
    )
    """
    json_data["endtime"] = str(datetime.now())
    for dataset_name in qa_datasets:
        json_data[f"{dataset_name}"] = {}
        exp_ans = expected_answers[dataset_name]
        pred_ans = predicted_answers[dataset_name]
        json_data[f"{dataset_name}"]["answered_questions"] = len(pred_ans)
        not_answered = unanswered_questions[dataset_name]
        total_questions = len(exp_ans) + not_answered
        accuracy = 0.0
        if exp_ans and pred_ans:  # make sure that both collections have values inside
            accuracy = accuracy_score(exp_ans, pred_ans)
        json_data[f"{dataset_name}"]["accuracy"] = accuracy
        elapsed_time = np.mean(question_times[dataset_name])/batch_size
        json_data[f"{dataset_name}"]["mqt"] = elapsed_time
        print(
            f"QA Dataset: {dataset_name} \n"
            f"Total Questions: {total_questions} \n"
            f"Answered Questions: {len(pred_ans)} \n"
            f"Accuracy: {accuracy} \n"
            f"Mean Querying Time: {elapsed_time} \n"
        )
        
        with open(f"{EXPERIMENT_DIR}/{filename}.json", "w") as f:
            json.dump(json_data, f, indent=1)