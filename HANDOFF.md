# HANDOFF

## 목표
레거시 x86-64/ARM용 C11 INT8/INT4 LLM 런타임 isa-lm. 순서: v0.1.0 배포 → Qwen2.5 → llama.cpp 비교 → 4-bit 개선 → prefill 속도.

## 완료 (모두 main에 merge)
- v0.1.0 릴리스(PR #14, tag `v0.1.0` = `3a1dc73`). 첫 Release 실패 원인: `scripts/smoke_release.sh`가 `set -e`에서 ISA guard의 exit 3으로 종료 → PR #15로 수정 후 tag 재생성
- Qwen2(PR #16), Q4 scale 탐색·llama.cpp 비교(PR #17)
- imatrix(PR #18): `ml_run --imatrix FILE`(채널별 제곱합 누적, 파일에 더함), `tools/convert.py --imatrix FILE`(q4 전용), `tools/mlb.py` `_q4_search(b, imp)`. 형식은 `docs/file-format.md` "Importance matrix file". `--imatrix` 없으면 변환 결과 비트 동일
- prefill 속도(PR #19): `kernels/x86_pack.h` `PK_TILES` 행 블록(`PK_ROW_BYTES` 512KB), AVX-512 VNNI matmul 두 행 그룹이 broadcast 공유, `ml_attention_batch`(위치×KV head를 스레드에 분배), prefill의 토큰별 연산 병렬화(`tokens_run`, `TOK_PARALLEL_MIN`). `ml_bench`에 matmul 측정 추가

## 남은 작업
1. imatrix 실측: Qwen2.5 0.5B Q8 변환 → wiki가 아닌 텍스트로 `perplexity.py --extra "--imatrix q.imat"` 수집 → `--wtype q4 --imatrix` 유무 perplexity 비교(기준: Q4 21.73, Q8 19.05) → `docs/user-guide.md`·README·CHANGELOG에 수치
2. `scripts/compare_llamacpp.sh` 재실행 → `docs/benchmarks.md` 표 갱신(현재 표는 PR #19 이전 값; "llama.cpp가 prefill 1.7–2.4배 빠름" 문구 포함)
3. 1스레드 prefill 추가 개선: 랜덤 모델 기준 llama.cpp(문서 표 Q8 335, Q4 365 tok/s) 대비 아직 느림. AVX-512 Q8 matmul은 이론치(약 560 GOPS)의 약 50%

## 측정 (PR #19, Qwen2.5-0.5B 모양 랜덤 모델 vocab 32000, 128토큰, 4 vCPU Xeon AVX-512 VNNI, tok/s)
| | PR #19 이전 | PR #19 |
|---|---:|---:|
| Q8 t1 / t4 prefill | 180 / 600 | 260 / 925 |
| Q4 t1 / t4 prefill | 250 / 600–730 | 290 / 865–918 |
- 모델 생성: `tools/gen_random.py --dim 896 --layers 24 --heads 14 --kv-heads 2 --hidden 4864 --vocab 32000 --max-seq 512`
- 커널: `build/avx512vnni-pt/ml_bench`(모델 없이) matmul_*_pk 행. 측정 편차 ±10%라 5회 중 최고값 비교

## 실패한 시도
- Q4 탐색 float64·21후보: 1B 변환 20분 이상 → float32 + 닫힌 형태 오차 + thread pool로 1B 약 75초
- float32 `.sum()`: CPU SIMD 폭에 따라 합산 순서가 달라져 CI와 로컬 모델이 다름 → `_sum32` 고정 순서. imatrix 정규화 평균도 `math.fsum`
- 테스트 seed: Q8 활성값 하나가 반올림 경계를 넘으면 이후 logits ~1% 변화. 양자화기를 바꾸면 `Makefile`의 mix/ctx seed를 다시 골라야 함
- prefetch는 Neoverse에서 손해(x86 전용), 토큰 tile 크기 변경은 이득 없음
- `ML_BATCH_MAX` 64: t1 +10%, t4 차이 없음 → 32 유지
- AVX-512 matmul에서 블록마다 xsum 뺄셈 제거(초기값으로 대체): 차이 없음
- AVX2 행 블록: 계산 병목이라 차이 없음(손해도 없음)

## 검증
```
make test e2e ISA=avx2
make e2e ISA=avx512vnni THREADS=pthread
make e2e ISA=neon_dotprod CROSS=aarch64-linux-gnu-
python3 -m pytest -q tests
make test-all
python3 tools/perplexity.py build/avx2/ml_run M.mlb T.mlt wiki.txt --ml-tok build/avx2/ml_tok --ctx 512 --chunks 4
```

## 환경
- HF 다운로드 가능(2026-10-26 확인: huggingface.co, us.aws.cdn.hf.co, Qwen2.5 0.5B 988MB 약 97MB/s). 이전 세션에서는 CDN이 차단됐었음
- cross 컴파일러(aarch64/armv7/mingw)와 pytest는 기본 미설치(`pip install pytest`). ARM·mingw는 PR CI로 확인
- 4 vCPU, L2 2MB/코어, vpdpbusd zmm 2개/사이클

## 주의
- 커밋 작성자 JonPark0 <yeochoony@gmail.com>, AI 흔적(trailer 등) 금지
- main 직접 push 금지. PR 생성·merge는 소유자가 함. 세션 지정 브랜치 외 push와 force push는 권한 판정에서 거부됨
- tag는 소유자가 push(PowerShell에서 `<...>` 자리표시자는 실제 값으로)
- 이 환경은 CI 로그·GraphQL 차단 → 실패 원인은 CI annotation(`Failure details` 단계)으로 확인. GitHub MCP `get_job_logs`로는 로그 tail을 볼 수 있음
- `pkill -f`는 자신의 셸을 죽임
