import os
import json
import argparse

from tqdm.auto import tqdm
from transformers import pipeline

from utils.common import prepare_dir
from utils.models import get_tokenizer, get_model

MODEL_CLASSES = ["koelectra-base-v2", "koelectra-small-v2", "koelectra-small-v3", "koelectra-base-v3", "multilingual-bert"]

def get_input_data(in_fn):
    in_data = []
    with open(in_fn, "r", encoding="utf-8") as f:
        in_data = json.load(f)

    context = in_data["context"]
    questions = in_data["questions"]

    return context, questions

def run_predict(pipeline, fns):
    in_fn = fns["input"]
    out_fn = fns["output"]

    context, questions = get_input_data(in_fn)

    out_data = {
        "context" : context,
        "qas" : []
    }

    for question in tqdm(questions):
        try:
            result = pipeline(context=context, question=question, handle_impossible_answer=True) # for impossible question
            # result = pipeline(context=context, question=question)

            out_data["qas"].append({
                "question" : question,
                "possible_score" : result["score"],
                "start" : result["start"],
                "end" : result["end"],
                "answer" : result["answer"]
            })
        except:
            continue

    with open(out_fn, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=4)
        print("Pre-processed data is dumped at ", out_fn)

if __name__ == '__main__':
    # Argument Setting -------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_type", type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES))

    # Other parameters
    parser.add_argument("--input_dir", type=str, default="./input",
                        help="path to input directory")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="path to output directory")

    args = parser.parse_args()
    # ------------------------------------------------------------------------------------------------------------------

    model_type = args.model_type.lower()
    tokenizer = get_tokenizer(model_type)
    model = get_model(model_type)

    mrc_pipeline = pipeline("question-answering", tokenizer=tokenizer, model=model)

    input_dir = args.input_dir
    output_dir = args.output_dir
    prepare_dir(output_dir)
    fns = {
        "input" : os.path.join(args.input_dir, "qg.json"),
        "output" : os.path.join(args.output_dir, "output.json")
    }

    run_predict(mrc_pipeline, fns)
