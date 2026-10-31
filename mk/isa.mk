# ISA selection: flags, fallback chain, target arch, emulator prefix.

ISAS := scalar sse2 sse41 avx2 avxvnni avx512vnni amx neon neon_dotprod neon_i8mm sve
ifeq ($(filter $(ISA),$(ISAS)),)
$(error Unknown ISA '$(ISA)'. Valid: $(ISAS))
endif

HOST_ARCH := $(shell uname -m | sed -e 's/^arm.*/arm/' -e 's/^arm64$$/aarch64/')
ARCH := $(shell $(CC) -dumpmachine | cut -d- -f1 | sed -e 's/^arm.*/arm/' -e 's/^i.86$$/x86/')

X86_ISAS := sse2 sse41 avx2 avxvnni avx512vnni amx
ARM_ISAS := neon neon_dotprod neon_i8mm sve
ifneq ($(filter $(ISA),$(X86_ISAS)),)
  ifneq ($(ARCH),x86_64)
    $(error ISA=$(ISA) needs an x86_64 compiler, got $(ARCH))
  endif
endif
ifneq ($(filter $(ISA),$(ARM_ISAS)),)
  ifeq ($(filter $(ARCH),arm aarch64),)
    $(error ISA=$(ISA) needs an ARM compiler (set CROSS=...), got $(ARCH))
  endif
endif
ifneq ($(filter $(ISA),neon_dotprod neon_i8mm sve),)
  ifneq ($(ARCH),aarch64)
    $(error ISA=$(ISA) needs AArch64)
  endif
endif

# Fallback chains, lowest first. avx512vnni skips avxvnni: AVX512-VNNI CPUs
# (Ice Lake, Cascade Lake) may lack the VEX AVX-VNNI encoding.
CHAIN_scalar       := scalar
CHAIN_sse2         := scalar sse2
CHAIN_sse41        := $(CHAIN_sse2) sse41
CHAIN_avx2         := $(CHAIN_sse41) avx2
CHAIN_avxvnni      := $(CHAIN_avx2) avxvnni
CHAIN_avx512vnni   := $(CHAIN_avx2) avx512vnni
CHAIN_amx          := $(CHAIN_avx512vnni) amx
CHAIN_neon         := scalar neon
CHAIN_neon_dotprod := $(CHAIN_neon) neon_dotprod
CHAIN_neon_i8mm    := $(CHAIN_neon_dotprod) neon_i8mm
CHAIN_sve          := $(CHAIN_neon_dotprod) sve
CHAIN := $(CHAIN_$(ISA))

ISAFLAGS_scalar       :=
ISAFLAGS_sse2         := -msse2
ISAFLAGS_sse41        := -msse4.1
ISAFLAGS_avx2         := -mavx2 -mfma
ISAFLAGS_avxvnni      := -mavx2 -mfma -mavxvnni
ISAFLAGS_avx512vnni   := -mavx512f -mavx512bw -mavx512vnni -mfma
ISAFLAGS_amx          := $(ISAFLAGS_avx512vnni) -mamx-tile -mamx-int8
ifeq ($(ARCH),arm)
  ISAFLAGS_neon       := -march=armv7-a -mfpu=neon -mfloat-abi=hard
else
  ISAFLAGS_neon       := -march=armv8-a
endif
ISAFLAGS_neon_dotprod := -march=armv8.2-a+dotprod
ISAFLAGS_neon_i8mm    := -march=armv8.2-a+dotprod+i8mm
ISAFLAGS_sve          := -march=armv8.2-a+dotprod+sve
ISAFLAGS := $(ISAFLAGS_$(ISA))

upper = $(shell echo $(1) | tr a-z A-Z)
ISADEFS := -DML_ISA_$(call upper,$(ISA)) $(foreach i,$(CHAIN),-DML_ENABLE_$(call upper,$(i)))

# Emulator prefix for cross builds.
QEMU_CPU ?= max
ifneq ($(ARCH),$(HOST_ARCH))
  ifeq ($(ARCH),aarch64)
    RUN ?= qemu-aarch64 -cpu $(QEMU_CPU)
  else ifeq ($(ARCH),arm)
    RUN ?= qemu-arm -cpu max
  endif
  CROSS_LDFLAGS := -static
  ARCHSUF := -$(ARCH)
endif
