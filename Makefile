# Legacy-ISA ML runtime.  make ISA=<isa> [THREADS=1|pthread] [ASM=0|1] [CROSS=prefix] [DEBUG=0|1]
ISA     ?= scalar
THREADS ?= 1
ASM     ?= 0
DEBUG   ?= 0
CROSS   ?=
CC      := $(CROSS)gcc
PYTHON  ?= python3

include mk/isa.mk

BUILD := build/$(ISA)$(ARCHSUF)$(if $(filter pthread,$(THREADS)),-pt)$(if $(filter 1,$(ASM)),-asm)

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
APPS  :=

.PHONY: all test clean guard-test
.SECONDARY:
all: $(addprefix $(BUILD)/,$(TESTS) $(APPS))

$(BUILD)/obj/%.o: %.c
	@mkdir -p $(dir $@)
	@$(CC) $(if $(filter $<,$(NOISA_SRC)),$(BASE_CFLAGS),$(CFLAGS)) -c $< -o $@

$(BUILD)/test_%: $(BUILD)/obj/tests/test_%.o $(GUARD_OBJ) $(LIB_OBJ)
	@$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS)

test: $(addprefix $(BUILD)/,$(TESTS))
	@set -e; $(foreach t,$(TESTS),$(RUN) $(BUILD)/$(t) $(TESTARGS_$(t));)

clean:
	rm -rf build

-include $(shell find build -name '*.d' 2>/dev/null)

GUARD_ISAS := scalar sse2 sse41 avx2 avxvnni avx512vnni
guard-test:
	@for i in $(GUARD_ISAS); do $(MAKE) -s ISA=$$i build/$$i/test_quant || exit 1; done
	@sh tests/test_isa_guard.sh
