# HANDOFF

## 목표
레거시 x86-64/ARM용 C11 INT8/INT4 LLM 런타임 isa-lm. 순서: v0.1.0 배포 → Qwen2.5 → llama.cpp 비교 → 4-bit 개선 → prefill 속도 → 지원 범위 확장.

## 완료 (main에 merge)
- v0.1.0 릴리스(PR #14, tag `v0.1.0` = `3a1dc73`). 첫 Release 실패 원인: `scripts/smoke_release.sh`가 `set -e`에서 ISA guard의 exit 3으로 종료 → PR #15로 수정 후 tag 재생성
- Qwen2(PR #16), Q4 scale 탐색·llama.cpp 비교(PR #17)
- imatrix(PR #18, #20, #21): `ml_run --imatrix FILE`(채널별 제곱합, 파일에 누적, PATH.tmp 후 rename), `tools/convert.py --imatrix FILE`(q4 전용). 가중치 `imp·sqrt(s²+x²)`(llama.cpp 방식), 공유(tied) embedding은 classifier 중요도 사용. `--imatrix` 없으면 변환 결과 비트 동일
- prefill 속도(PR #19): `PK_TILES` 행 블록(`PK_ROW_BYTES` 512KB), AVX-512 matmul 두 행 그룹 broadcast 공유, `ml_attention_batch`(위치×head 묶음을 스레드에 분배, `ml_attention_batch_heads`로 scratch 상한), 토큰별 연산 병렬화(`tokens_run`)
- llama.cpp 비교 재측정(PR #21, `docs/benchmarks.md`), 외부 에이전트 검토 반영(PR #21)
- PR #22: 블록 합 psadbw(`kernels/x86_pack.h` `pk_block_sums`; Qwen2.5 0.5B 1스레드 prefill Q4 240→287, Q8 231→246 tok/s), VNNI matmul 내부 루프 펼침(`ML_UNROLL`: GCC가 `vpdpbusd` 누적 레지스터마다 복사 2개를 넣던 것 제거; 커널 AVX-512 Q8 +6~18%, AVX-VNNI Q8 +12~51%). AVX2·SSE4.1은 펼쳐도 이득 없어 그대로

## 진행 중 (브랜치 `claude/wonderful-allen-pmaeca`, 미merge)
- 자동 릴리스(`2ad27c2`): `.github/workflows/auto-release.yml`(main push마다 `scripts/release_check.sh`로 `ML_VERSION`의 tag 유무 확인 → 없으면 `release.yml`을 `workflow_call`로 호출, 게시할 때 그 커밋에 tag 생성). 버전만 올리고 CHANGELOG `## X.Y.Z` 없으면 PR CI의 "Release version check" 단계에서 실패. 실제 게시는 다음 버전 올릴 때 처음 확인됨
- Google 인증 태그(`aed56ea`)는 소유자 요청으로 되돌림(`7356037`). 다시 넣지 말 것

## Imatrix 측정 (Qwen2.5 0.5B, wikitext-2 test, imatrix = train 32K토큰)
| 가중치 | 4 chunk | 16 chunk | `--llama-cpp` |
|---|---:|---:|---:|
| Q8 | 19.01 | 22.10 | 13.33 |
| Q4 | 21.72 | 26.47 | 15.13 |
| Q4 imatrix | 20.92 | 25.30 | 14.52 |
- TinyStories imatrix: 16 chunk 25.74. imatrix 없이 sqrt(s²+x²): 26.67(나쁨). level 재반올림: 이득 작아 미적용

## 남은 작업
1. 1스레드 Q8 prefill: llama.cpp 361 대비 약 246 tok/s(PR #22 루프 펼침 전 측정, 이후 실제 모델 재측정은 컨테이너 잡음 ±15%로 미확정). AVX-512 Q8 커널 약 320~365 GOPS(이론 약 560): 블록마다 sub·cvt·mul·fma 4개가 남음. 세 행 그룹(NG=3): 결과 엇갈림으로 미적용. 커널 asm은 `objdump -d` 후 `vpdpbusd` 주변 `vmovdqa` 개수로 레지스터 복사 확인
2. 지원 범위 확장(아래 "지원 범위 계획")

## 지원 범위 계획 (2026-10-26 HF config 확인)
우선순위(가치/비용):
1. GPT-2 계열 ByteLevel pre-tokenizer(`Digits(individual)` + `ByteLevel(use_regex)`): SmolLM2(135M/360M/1.7B, `LlamaForCausalLM`, 구조는 이미 지원)가 tokenizer만으로 막힘(`convert_tokenizer.py`: "byte-level pre-tokenizer other than Split(regex) + ByteLevel"). GPT-2 정규식 변형 + 한 자리 숫자 분리만 추가하면 됨. `tests/check_tokenizer.py`로 `tokenizers`와 대조
2. Qwen3(0.6B~): Qwen2 + head별 QK RMSNorm, qkv bias 없음, `head_dim`(128)이 `dim/heads`(64)와 다름. `.mlb` 헤더에 head_dim·qk_norm 필드, `head_dim` 분리(현재 `convert.py`가 거부). tokenizer는 Qwen2와 같아 지원됨
3. RoPE scaling: linear(쉬움), YaRN(Qwen2.5 32K 초과 선택 사항). longrope(Phi-3)는 Phi-3 지원 때
4. Gemma 3 1B: GeGLU(gelu_tanh), (1+w) RMSNorm, pre/post norm, embedding×sqrt(dim), head_dim 256, query_pre_attn_scalar, 5:1 local(512)/global sliding window, 새 tokenizer 변형(Split on space). 비용 큼
5. Sliding window: Gemma 3에 필요(층별 local/global). Mistral v0.1은 지금처럼 문맥을 window로 제한하면 정확
- 범위 밖 유지: WordPiece·Unigram(인코더/T5 계열, 디코더 LLM 런타임에 해당 모델 거의 없음), MoE
- 원칙: 지원하지 않는 것은 지금처럼 변환기가 거부(틀린 모델을 쓰지 않음)

## 데이터
- wikitext-2: `Salesforce/wikitext` `wikitext-2-raw-v1/{test,train}-00000-of-00001.parquet`의 `text` 열을 그대로 이어 붙임(빈 행은 빈 문자열). 이 텍스트로 문서 수치(Q4 21.72) 재현됨. `pip install pyarrow` 필요
- 1 chunk = 512토큰, `--chunks 16` 평가가 4 chunk보다 안정적

## 실패한 시도
- Q4 탐색 float64·21후보: 1B 변환 20분 이상 → float32 + 닫힌 형태 오차 + thread pool로 1B 약 75초
- float32 `.sum()`: CPU SIMD 폭에 따라 합산 순서가 달라져 CI와 로컬 모델이 다름 → `_sum32` 고정 순서. imatrix 정규화 평균도 `math.fsum`
- 테스트 seed: Q8 활성값 하나가 반올림 경계를 넘으면 이후 logits ~1% 변화. 양자화기를 바꾸면 `Makefile`의 mix/ctx seed를 다시 골라야 함
- prefetch는 Neoverse에서 손해(x86 전용), 토큰 tile 크기 변경은 이득 없음
- `ML_BATCH_MAX` 64: t1 +10%, t4 차이 없음 → 32 유지
- AVX-512 matmul에서 블록마다 xsum 뺄셈 제거(초기값으로 대체): 차이 없음
- AVX2 행 블록: 계산 병목이라 차이 없음(손해도 없음)
- AVX-512 Q8 matmul 세 행 그룹: 결과 엇갈림

## 측정 방법
- 랜덤 모델: `tools/gen_random.py --dim 896 --layers 24 --heads 14 --kv-heads 2 --hidden 4864 --vocab 32000 --max-seq 512`
- 실제 모델: Qwen2.5-0.5B-Instruct `convert.py hf ... --max-seq 4096`, `ml_bench MODEL -t 1 -n 128`
- 커널: `build/avx512vnni-pt/ml_bench`(모델 없이) matmul_*_pk 행. 편차 ±10%라 이전/새 빌드를 번갈아 5회, 최고값 비교(이전 빌드는 `git worktree`로)
- llama.cpp 비교: b11835 빌드 4종(`docs/benchmarks.md` Reproducing), `qwen2.5-0.5b-instruct-fp16.gguf`(Qwen/Qwen2.5-0.5B-Instruct-GGUF)

## 외부 검토 (baze, agy)
- `baze claude --model <id> -p --permission-mode default --allowedTools ...`(읽기·make·python3·git 읽기만). `--dangerously-skip-permissions`는 Claude Code 하네스에서 권한 판정 거부
- agy(Gemini): `agy --dangerously-skip-permissions --sandbox --effort high -p ...`가 동작. 허용 목록 방식은 복합 셸 명령 하나에 세션 종료
- 보고서 파일 쓰기가 막히면 stdout으로 받음. 각 검토자는 별도 `git worktree`
- 비용: gpt-6-astra 1건 약 18,500, claude-fable-5-1 1건 약 10,300 크레딧(월 30,000, 매월 1일 갱신). 한도를 미리 정할 것

## PR 전 기본 검토: sait-3-pro (ainu 게이트웨이, 무료)
- 절차: PR용 커밋이 끝나면 아래 스크립트를 저장소 루트에서 `python3 sait_review.py origin/main HEAD` → 지적마다 코드로 확인·재현 → 실제 결함만 고침 → 결과를 소유자에게 보고(오탐은 이유와 함께)
- 오탐이 흔함: 첫 사용(자동 릴리스 diff)에서 지적 2개 모두 오탐(`action-gh-release`는 tag가 이미 있으면 `target_commitish`를 쓰지 않음, `workflow_call`의 `tag`는 필수라 `github.ref_name`으로 가지 않음). 그대로 믿지 말 것
- 게이트웨이: `$AINU_GATEWAY_URL`(OpenAI 호환 `/v1/chat/completions`, `/v1/models`), `$AINU_API_KEY`. 모델 `sait-3-pro`(200KB 입력 정확, 약 10~15초), `gpt-5-nano`(60KB부터 지시 무시·입력 잘림 → 짧은 질문만)
- 한계: 도구 호출 미지원(에이전트 하네스 불가, gpt-5-nano는 테스트 결과를 지어냄), 호출마다 약 1만 2천 토큰이 덧붙음, Python 기본 User-Agent는 403(헤더 지정 필요), 학교 서비스(ainu.inu.ac.kr) 웹 세션 경유라 대량 호출 자제
- `ainu` CLI는 새 세션의 setup script에서 설치됨(이 HANDOFF를 쓴 세션에는 없었음, curl/Python으로 직접 호출)

```python
"""Single-shot code review by sait-3-pro through the ainu gateway.
usage: python3 sait_review.py BASE [HEAD]   (run in the repo; prints the review)
Sends `git diff BASE..HEAD` plus the full text of every changed file (up to
~180 KB), and asks for concrete defects only."""
import json, os, subprocess, sys, urllib.request

base, head = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "HEAD")
git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout
diff = git("diff", f"{base}..{head}")
files = [f for f in git("diff", "--name-only", f"{base}..{head}").split() if f]
ctx = ""
for f in files:
    try:
        text = git("show", f"{head}:{f}")
    except subprocess.CalledProcessError:
        continue   # deleted
    if len(ctx) + len(text) > 120_000:
        ctx += f"\n// FILE {f}: omitted (size)\n"
        continue
    ctx += f"\n// FILE {f}\n{text}"
log = git("log", "--format=%h %s%n%b", f"{base}..{head}")
prompt = f"""You are reviewing a change to isa-lm, a C11/Python INT8/INT4 LLM inference runtime
for x86-64 and ARM (compile-time ISA kernels, bit-identical results across ISAs, thread counts and
batch sizes), with GitHub Actions workflows and Jekyll docs.

Report only concrete defects the change introduces: wrong results, crashes, races, out-of-bounds
access, portability breaks (ARM, Windows/MinGW, 32-bit), workflow logic errors, wrong docs claims.
For each: file:line, what goes wrong, the concrete input or scenario, and a minimal fix. Say
"No defects found" if none. No style comments, no praise, no summary of the change.

Commit messages:
{log}

Diff:
```diff
{diff[:60000]}
```

Full text of the changed files after the change:
{ctx}
"""
req = urllib.request.Request(
    os.environ["AINU_GATEWAY_URL"].rstrip("/") + "/v1/chat/completions",
    data=json.dumps({"model": "sait-3-pro", "messages": [{"role": "user", "content": prompt}]}).encode(),
    headers={"Authorization": "Bearer " + os.environ["AINU_API_KEY"], "Content-Type": "application/json",
             "User-Agent": "isa-lm-review/1"})   # the gateway refuses Python-urllib's
with urllib.request.urlopen(req, timeout=600) as r:
    print(json.load(r)["choices"][0]["message"]["content"])
```


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
- HANDOFF는 `handoff` 브랜치에만 둠(main에 넣지 않음). 세션은 지정 브랜치에 HANDOFF 커밋 → 소유자가 `git push -f origin <커밋>:handoff` → 세션이 삭제 커밋

## 주의
- 커밋 작성자 JonPark0 <yeochoony@gmail.com>, AI 흔적(trailer 등) 금지
- main 직접 push 금지. PR 생성·merge는 소유자가 함. 세션 지정 브랜치 외 push와 force push는 권한 판정에서 거부됨
- tag는 소유자가 push(PowerShell에서 `<...>` 자리표시자는 실제 값으로)
- 이 환경은 CI 로그·GraphQL 차단 → 실패 원인은 CI annotation(`Failure details` 단계)으로 확인. GitHub MCP `get_job_logs`로는 로그 tail을 볼 수 있음
- `pkill -f`는 자신의 셸을 죽임
