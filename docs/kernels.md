# Writing kernels

## Add a kernel for an existing ISA

1. Implement the kernel in `kernels/<isa>/k_<isa>.c` with the exact signature
   of its scalar version in `kernels/scalar/k_scalar.h`, renamed to
   `ml_k_<op>_<isa>`.
2. In `kernels/<isa>/k_<isa>.h`, declare it and add
   `#define ML_HAS_<OP>_<ISA>`.
3. Run `make ISA=<isa> test e2e`.

You don't need to touch anything else: `config/ml_config.h` binds the new kernel
automatically, and every ISA higher in the chain that lacks the op now inherits
it.

## Add a new ISA

1. Add it to `mk/isa.mk`:
   - its name to `ISAS`;
   - a `CHAIN_<isa>` that lists the ISAs it falls back to;
   - its compiler flags in `ISAFLAGS_<isa>`;
   - an architecture check, if the ISA only exists on x86 or only on ARM.
2. Create `kernels/<isa>/k_<isa>.h`. It may start empty.
3. Add the ISA to `config/ml_config.h`, in both places:
   - include `kernels/<isa>/k_<isa>.h` under `#ifdef ML_ENABLE_<ISA>`;
   - put its `ML_HAS_*_<ISA>` test in each op's `#if` chain, in priority order.
4. In `core/isa_check.c`, add the CPUID/HWCAP bits the new flags require.
5. Add negative guard cases to `tests/test_isa_guard*.sh`: a qemu CPU model that
   lacks the feature must make the binary exit with code 3.
6. Add the target to the `CMakeLists.txt` flag table if it is x86, and add a
   row to `tests/test_all.sh`.

## Rules every kernel must follow

| Rule | Why |
|------|-----|
| Handle any `n`, any row range `[r0, r1)`, rows not a multiple of 4, and an odd number of 32-blocks. Use scalar tails or masked or predicated loads. | `test_kernels` runs n ∈ {1, 7, 32, 288, 4097}, rows ∈ {1, 3, 4, 7, 64}, cols ∈ {32, 96, 256}, and `r0` ∈ {0, 1}. |
| Leave rows outside `[r0, r1)` untouched. | Threads write disjoint row ranges of the same output. |
| `quantize_q8` must be **bit-identical** to scalar: compute `id = 127/amax` in scalar code, then a plain multiply and add (no FMA) of `copysign(0.5, x)`, then truncate, then clamp to ±127. Blocks with `amax < ML_QUANT_TINY` are all zero. | Activations must match across ISAs, so e2e tests can compare generated text against the scalar build. |
| Integer dot products must never saturate. | Quants are in [-127, 127], which keeps `pmaddubsw` pair sums ≤ 32258. |
| Use unaligned loads. | Tensor arrays are 64-byte aligned, but row and block offsets are not. |
| Use only intrinsics from the ISA's own flags, or from ISAs those flags imply. | The ISA guard only checks what `mk/isa.mk` enables. |
| No allocation and no global state. Small stack buffers are fine. | Kernels run on several threads at once. |

## Integer dot-product idioms per ISA

| ISA | Q8 × Q8, 32 elements |
|-----|----------------------|
| `sse2` | Sign-extend both operands to int16 (`punpck` + `psraw 8`), then `pmaddwd`. |
| `sse41`, `avx2` | `pmaddubsw(abs(x), sign(w, x))`, then `pmaddwd(·, 1)`. |
| `avxvnni` | `vpdpbusd(0, abs(x), sign(w, x))`. |
| `avx512vnni` | `vpdpbusd` on two blocks at once. There is no 512-bit `psignb`, so negate `w` with `mask_sub` where `x < 0`. |
| `neon` | `vmull_s8` + `vpadalq_s16` (works on ARMv7 and AArch64). |
| `neon_dotprod` | `vdotq_s32`. |
| `sve` | `svdot_s32` over `svcntb()` bytes. Per-block scales come from a gather, so any vector length works. |

Q4 kernels unpack nibbles first (`& 0x0F`, `>> 4`, subtract 8) and then reuse
the Q8 path.

The matvecs process **4 rows per pass**, so each activation block is loaded once
and used four times. They convert int32 lanes to float, multiply by the block
scale `d_w · d_x`, and reduce horizontally only once per row. With that
deferred reduction, precision is bounded by the magnitude of the products, and
`test_kernels` measures error against exactly that scale.

## Packed kernels

ISAs with packed kernels implement four more functions over the layout in
[`core/pack.h`](../core/pack.h):

```c
void ml_k_matvec_q8_pk_<isa>(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1);
void ml_k_matvec_q4_pk_<isa>(float *y, const ml_qtensor *W, const ml_qact *x, int r0, int r1);
void ml_k_matmul_q8_pk_<isa>(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1);
void ml_k_matmul_q4_pk_<isa>(float *Y, int ldy, const ml_qtensor *W, const ml_qbatch *X, int r0, int r1);
```

`W->pq` and `W->pd` hold the packed quants and scales. `r0` is a multiple of
the group size, and rows at or above `r1` must not be written.

To register them, define the following in the ISA header, then add a branch
for the ISA to both packed `#if` chains in `config/ml_config.h`:

- `ML_HAS_MATVEC_PK_<ISA>` and, if the matmul exists, `ML_HAS_MATMUL_PK_<ISA>`;
- the group size `ML_PACK_G`, the Q8 offset `ML_PACK_Q8_OFFSET` (128 if the dot
  instruction needs unsigned weights), and `ML_PACK_Q8`, set to 0 to leave Q8
  row-major.

An ISA with a packed matvec but no matmul falls back to `ml_k_matmul_*_pk_loop`.

The tests hold packed kernels to these rules:

- **Matvec:** must match the scalar row-major kernel. It is bit-exact with unit
  scales, and within 1e-5 of the product magnitude otherwise.
- **Matmul:** each token's column must equal that ISA's packed matvec **bit for
  bit**. Keep the float operations in the same order: the same multiply, then
  the same add or FMA.

## Replacing a kernel with assembly

1. List the kernel's name in `kernels/<isa>/asm.mk`: `ASM_<isa> := add`.
2. Write `kernels/<isa>/<name>.S` exporting `ml_k_<name>_<isa>`. Use the same C
   ABI: System V x86-64, AAPCS or AAPCS64.
3. Wrap the C version in `#ifndef ML_ASM_<NAME>_<ISA>`.
4. Build with `make ISA=<isa> ASM=1`.

`kernels/sse2/add.S` is the sample. To check that the assembly is in use, run
`objdump -d build/sse2-asm/test_kernels | grep -A4 '<ml_k_add_sse2>:'`.
