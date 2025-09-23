import json, tqdm, torch, faiss, numpy as np
from src.medrag import MedRAG
from src.mirage import *
import os
import argparse

def evaluate(
    dataset_name,
    medrag,
    method="rag",
    llm_name="meta-llama/Meta-Llama-3-8B-Instruct",
    save_dir=""
):
    correct = 0
    total = 0
    predictions = []

    for q_idx in range(len(dataset_name)):
        item = dataset_name[q_idx]
        question = item["question"]
        options = item["options"]
        correct_answer = item["answer"]

        generated_answer, snippets, scores = medrag.answer(
            question=question,
            options=options,
            save_dir=save_dir,
            file_prefix=dataset_name.index[q_idx],
        )
        predicted_answer = locate_answer(generated_answer)
        predictions.append(predicted_answer)

        if predicted_answer == correct_answer:
            correct += 1
        total += 1

    acc = correct / total
    std = np.sqrt(acc * (1 - acc) / total)
    return acc, std


def run_experiments(
    llm_name,
    k,
    corpus_name,
    retriever_name,
    method,
    qadatasets,
):
    benchmark = json.load(open("benchmark.json"))
    answer_list = ["A", "B", "C", "D"]
    answer2idx = {ans: i for i, ans in enumerate(answer_list)}
    results = {}

    if (method.lower() == "rag"):
        medrag = MedRAG(
            llm_name=llm_name,
            rag=True,
            retriever_name=retriever_name,
            corpus_name=corpus_name,
            corpus_cache=False,
        )
        save_dir = os.path.join("../prediction", ds_str, f"rag_{k}", llm_name, corpus_name, retriever_name)
    else:
        medrag = MedRAG(llm_name=llm_name, rag=False)
        save_dir = os.path.join("../prediction", ds_str, "cot", llm_name)
    print("saving @", save_dir)
    os.makedirs(save_dir, exist_ok=True)

    for qads in qadatasets:
        ds_str = qads
        qadataset = MirageQA(qads)

        multi_choice_acc, multi_choice_std = evaluate(
            qadataset,
            medrag,
            method=method,
            llm_name=llm_name,
            save_dir =save_dir
        )
        print(qads, multi_choice_acc, multi_choice_std)
        results[qads] = {"accuracy": multi_choice_acc, "std": multi_choice_std}

    print(results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MedRAG QA experiments")

    parser.add_argument("--llm_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="LLM model name")
    parser.add_argument("--k", type=int, default=8,
                        help="Number of retrieved documents")
    parser.add_argument("--corpus_name", type=str, default="PubMed",
                        help="Corpus to use for retrieval")
    parser.add_argument("--retriever_name", type=str, default="Contriever",
                        help="Retriever model name")
    parser.add_argument("--method", type=str, default="rag",
                        choices=["rag", "cot"],
                        help="Method to use (rag or cot)")
    parser.add_argument("--qadatasets", type=str, nargs="+", default=["bioasq", "pubmedqa"],
                        help="List of QA datasets")

    args = parser.parse_args()

    run_experiments(
        llm_name=args.llm_name,
        k=args.k,
        corpus_name=args.corpus_name,
        retriever_name=args.retriever_name,
        method=args.method,
        qadatasets=args.qadatasets,
    )
