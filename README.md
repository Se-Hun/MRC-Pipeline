# MRC-Pipeline
Transformers Pipeline about MRC(Question Answering).

At this project, Transformers Pipelines are trained by KorQuAD(The Korean Question Answering Dataset).

## Available Pipeline

| Model                   | Link            |
| ----------------------- | --------------- |
| koelectra-base-v2       | [koelectra-base-v2-finetuned-korquad](https://huggingface.co/monologg/koelectra-base-v2-finetuned-korquad)              | 
| koelectra-small-v2      | [koelectra-small-v2-distilled-korquad-384](https://huggingface.co/monologg/koelectra-small-v2-distilled-korquad-384)              | 

## Requirements

* torch >= 1.4.0
* transformers == 3.0.2

## Usage

### 입력 데이터 파일 준비

* 다음과 같은 형태의 JSON 파일을 준비하고 파일명을 `qg.json`으로 수정
* JSON 파일은 보편적인 MRC Q/A Task의 입력인 context와 question으로 구성된다.

```json
{
  "context" : "단락 단위의 문맥 정보",
  "questions" : [
    "질문 1",
    "질문 2",
    ...
  ]
}
```

### 실행

```bash
$ python run_mrc_pipeline.py --model_type MODEL_TYPE --input_dir INPUT_DIR --output_dir OUTPUT_DIR
```

* `MODEL_TYPE` : 지원하는 모델의 이름을 지정
* `INPUT_DIR` : 준비한 입력 데이터 파일이 있는 폴더의 경로를 지정
* `OUTPUT_DIR` : 결과를 저장할 폴더의 경로를 지정

## TODO list

- [ ] 다른 transformer 기반의 모델들 추가


## References

- [KorQuAD](https://korquad.github.io/)
- [Huggingface Transformers : Pipelines](https://huggingface.co/transformers/main_classes/pipelines.html)
- [monologg/KoELECTRA](https://github.com/monologg/KoELECTRA)
- [monologg/KoELECTRA-Pipeline](https://github.com/monologg/KoELECTRA-Pipeline)