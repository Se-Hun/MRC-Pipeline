# MRC-Pipeline
Transformers Pipeline about MRC(Question Answering).

## Available Pipeline

| Model                   | Link            |
| ----------------------- | --------------- |
| koelectra-base-v2       | [koelectra-base-v2-finetuned-korquad](https://huggingface.co/monologg/koelectra-base-v2-finetuned-korquad)              | 
| koelectra-small-v2      | [koelectra-small-v2-distilled-korquad-384](https://huggingface.co/monologg/koelectra-small-v2-distilled-korquad-384)              | 

## Requirements

* torch >= 1.4.0
* transformers == 3.0.2

## Usage

```bash
$ python run_mrc_pipeline.py
```

## TODO list

- [ ] 다른 transformer 기반의 모델들 추가


## References

- [Huggingface Transformers : Pipelines](https://huggingface.co/transformers/main_classes/pipelines.html)
- [monologg/KoELECTRA](https://github.com/monologg/KoELECTRA)
- [monologg/KoELECTRA-Pipeline](https://github.com/monologg/KoELECTRA-Pipeline)