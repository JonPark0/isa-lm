# HANDOFF

## 목표
레거시 x86-64/ARM용 C11 INT8/INT4 LLM 런타임 isa-lm. 순서: v0.1.0 배포 → Qwen2.5 → llama.cpp 비교 → 4-bit 개선.

## 완료
- `release/v0.1.0` (PR #14, CI 통과): `scripts/package_release.sh`, `smoke_release.sh`, `.github/workflows/release.yml`
- `feat/qwen2`: q/k/v bias(.mlb offset 76), NFC·한 자리 숫자 tokenizer(MLT1 flag 8/16), `--chat qwen|chatml`, 변환기 모델 검사
- `feat/q4-bench`: Q4 scale 탐색(`tools/mlb.py` `_q4_search`), `scripts/compare_llamacpp.sh`, `docs/benchmarks.md`
- 세 브랜치는 이 순서로 쌓여 있음

## 남은 작업
1. PR #14 merge → `v0.1.0` tag push → feat/qwen2, feat/q4-bench 순서로 PR(base main)
2. imatrix: `model/transformer.c` `ml_forward`/`prefill_step`에서 행렬 입력의 채널별 제곱합 누적, `app/ml_run.c`에 출력 옵션, `tools/mlb.py` `_q4_search`의 가중치 `w`에 채널 중요도 곱하기, `tools/convert.py`에 `--imatrix`
3. AVX-512 prefill(llama.cpp 대비 1.7–2.4배 느림): `kernels/avx512vnni/k_avx512vnni.c` matmul tile, `ML_BATCH_MAX`(model/transformer.h)

## 실패한 시도
- Q4 탐색 float64·21후보: 1B 변환 20분 이상 → float32 + 닫힌 형태 오차 + thread pool로 1B 약 75초
- float32 `.sum()`: CPU SIMD 폭에 따라 합산 순서가 달라져 CI와 로컬 모델이 다름 → `_sum32` 고정 순서
- 테스트 seed: Q8 활성값 하나가 반올림 경계를 넘으면 이후 logits ~1% 변화. 양자화기를 바꾸면 `Makefile`의 mix/ctx seed를 다시 골라야 함
- prefetch는 Neoverse에서 손해(x86 전용), tile 크기 변경은 이득 없음

## 검증
```
make test e2e ISA=avx2
make e2e ISA=neon_dotprod CROSS=aarch64-linux-gnu-
python3 -m pytest -q tests
make test-all
python3 tools/perplexity.py build/avx2/ml_run M.mlb T.mlt wiki.txt --ml-tok build/avx2/ml_tok --ctx 512 --chunks 4
```

## 주의
- 커밋 작성자 JonPark0 <yeochoony@gmail.com>, AI 흔적(trailer 등) 금지
- main 직접 push 금지. PR 생성·merge는 소유자가 함
- 이 환경은 CI 로그·GraphQL 차단 → 실패 원인은 CI annotation(`Failure details` 단계)으로 확인
- `pkill -f`는 자신의 셸을 죽임
