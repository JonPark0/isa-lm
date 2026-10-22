# Legacy-ISA ML runtime.  make ISA=<isa> [THREADS=1|pthread] [ASM=0|1] [CROSS=prefix] [DEBUG=0|1]
ISA     ?= scalar
THREADS ?= 1
ASM     ?= 0
DEBUG   ?= 0
CROSS   ?=
CC      := $(CROSS)gcc
PYTHON  ?= python3

include mk/isa.mk

BUILD := build/$(ISA)$(ARCHSUF)$(if $(filter pthread,$(THREADS)),-pt)$(if $(filter 1,$(ASM)),-asm)$(if $(filter 1,$(DEBUG)),-dbg)
# Extra flags get their own object dir so objects never mix flag sets.
ifneq ($(strip $(EXTRA_CFLAGS)),)
  BUILD := $(BUILD)-x$(shell echo '$(EXTRA_CFLAGS)' | cksum | cut -d' ' -f1)
endif

BASE_CFLAGS := -std=c11 -Wall -Wextra -Werror -ffp-contract=off -D_GNU_SOURCE -I. -MMD -MP
ifeq ($(DEBUG),1)
  BASE_CFLAGS += -O0 -g
else
  BASE_CFLAGS += -O2
endif
BASE_CFLAGS += $(ISADEFS) $(EXTRA_CFLAGS)
CFLAGS  := $(BASE_CFLAGS) $(ISAFLAGS)
LDFLAGS := $(CROSS_LDFLAGS) $(EXTRA_LDFLAGS)
LDLIBS  := -lm

LIB_SRC := $(wildcard core/*.c) \
           $(foreach i,$(CHAIN),$(wildcard kernels/$(i)/*.c)) \
           $(wildcard ops/*.c) $(wildcard model/*.c)
# Compiled without ISA flags: these run before the ISA guard passes.
NOISA_SRC := core/isa_check.c app/main_guard.c
LIB_OBJ := $(LIB_SRC:%.c=$(BUILD)/obj/%.o)
GUARD_OBJ := $(BUILD)/obj/app/main_guard.o

TESTS := $(patsubst tests/%.c,%,$(wildcard tests/test_*.c))
APPS  := ml_run

.PHONY: all test clean guard-test e2e scalar-ref
.SECONDARY:
all: $(addprefix $(BUILD)/,$(TESTS) $(APPS))

$(BUILD)/obj/%.o: %.c
	@mkdir -p $(dir $@)
	@$(CC) $(if $(filter $<,$(NOISA_SRC)),$(BASE_CFLAGS),$(CFLAGS)) -c $< -o $@

$(BUILD)/test_%: $(BUILD)/obj/tests/test_%.o $(GUARD_OBJ) $(LIB_OBJ)
	@$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS)

MODELS := build/models/rand_q8.mlb build/models/rand_q4.mlb
build/models/rand_%.mlb: tools/gen_random.py tools/mlb.py
	@mkdir -p $(dir $@)
	@$(PYTHON) tools/gen_random.py --out $@ --wtype $* --dump-emb-row 5

TESTARGS_test_load := $(MODELS)
TOKENIZER := build/models/tok96.bin
$(TOKENIZER): tools/gen_tokenizer.py
	@mkdir -p $(dir $@)
	@$(PYTHON) tools/gen_tokenizer.py --vocab 96 --out $@

$(BUILD)/ml_run: $(BUILD)/obj/app/ml_run.o $(GUARD_OBJ) $(LIB_OBJ)
	@$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS)

# Host scalar build: the reference e2e compares generated text against.
scalar-ref:
	@$(MAKE) -s ISA=scalar CROSS= THREADS=1 ASM=0 DEBUG=0 EXTRA_CFLAGS= RUN= build/scalar/ml_run

e2e: $(BUILD)/ml_run $(MODELS) $(TOKENIZER) scalar-ref
	@sh tests/test_e2e.sh $(ISA) $(BUILD) "$(RUN)"

test: $(addprefix $(BUILD)/,$(TESTS)) $(MODELS)
	@set -e; $(foreach t,$(TESTS),$(RUN) $(BUILD)/$(t) $(TESTARGS_$(t));)

clean:
	rm -rf build

-include $(shell find build -name '*.d' 2>/dev/null)

GUARD_ISAS := scalar sse2 sse41 avx2 avxvnni avx512vnni
guard-test:
	@for i in $(GUARD_ISAS); do $(MAKE) -s ISA=$$i build/$$i/test_quant || exit 1; done
	@sh tests/test_isa_guard.sh
