from transformers import ElectraTokenizer, ElectraForQuestionAnswering, pipeline

def get_tokenizer(model_type):
    if model_type == "koelectra-base-v2":
        tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v2-finetuned-korquad")
        return tokenizer

    elif model_type == "koelectra-small-v2":
        tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-v2-distilled-korquad-384")
        return tokenizer

    else:
        raise ValueError(model_type)


def get_model(model_type):
    if model_type == "koelectra-base-v2":
        model = ElectraForQuestionAnswering.from_pretrained("monologg/koelectra-base-v2-finetuned-korquad")
        return model

    elif model_type == "koelectra-small-v2":
        model = ElectraForQuestionAnswering.from_pretrained("monologg/koelectra-small-v2-distilled-korquad-384")
        return model

    else:
        raise ValueError(model_type)