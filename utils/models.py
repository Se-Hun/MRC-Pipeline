from transformers import ElectraTokenizer, ElectraForQuestionAnswering, BertTokenizer, BertForQuestionAnswering

def get_tokenizer(model_type):
    if model_type == "koelectra-small-v2":
        tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-v2-distilled-korquad-384")
        return tokenizer

    elif model_type == "koelectra-base-v2":
        tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v2-finetuned-korquad")
        return tokenizer

    elif model_type == "koelectra-small-v3":
        tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-v3-finetuned-korquad")
        return tokenizer

    elif model_type == "koelectra-base-v3":
        tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-finetuned-korquad")
        return tokenizer

    elif model_type == "multilingual-bert":
        tokenizer = BertTokenizer.from_pretrained("sangrimlee/bert-base-multilingual-cased-korquad")
        return tokenizer

    else:
        raise ValueError(model_type)


def get_model(model_type):
    if model_type == "koelectra-small-v2":
        model = ElectraForQuestionAnswering.from_pretrained("monologg/koelectra-small-v2-distilled-korquad-384")
        return model

    elif model_type == "koelectra-base-v2":
        model = ElectraForQuestionAnswering.from_pretrained("monologg/koelectra-base-v2-finetuned-korquad")
        return model

    elif model_type == "koelectra-small-v3":
        model = ElectraForQuestionAnswering.from_pretrained("monologg/koelectra-small-v3-finetuned-korquad")
        return model

    elif model_type == "koelectra-base-v3":
        model = ElectraForQuestionAnswering.from_pretrained("monologg/koelectra-base-v3-finetuned-korquad")
        return model

    elif model_type == "multilingual-bert":
        tokenizer = BertForQuestionAnswering.from_pretrained("sangrimlee/bert-base-multilingual-cased-korquad")
        return tokenizer

    else:
        raise ValueError(model_type)