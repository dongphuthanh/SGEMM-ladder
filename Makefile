NVCC  ?= nvcc
ARCH  ?= sm_120
FLAGS := -O2 -arch=$(ARCH)
BUILD := build

# Layout:
#   ladder/     the nine SGEMM kernels, one per rung (each standalone)
#   attention/  gemm.cuh, softmax.cu, naive_attn.cu, online_attn.cu
#   test/       the harnesses; each #includes the kernel files it tests
#   build/      binaries (created by make, not tracked)
#
# The harnesses #include the kernel sources, so the kernel files are listed as
# prerequisites: editing any kernel re-triggers the build of its harness.

LADDER := $(addprefix ladder/, naive.cu sharedmemtile.cu threadtiling.cu sharedthreadtile.cu \
            sharedthreadtilev2.cu transpose.cu doublebuffer.cu bk16.cu warptiling.cu)
ATTN   := $(addprefix attention/, gemm.cuh softmax.cu naive_attn.cu online_attn.cu)

$(BUILD):
	mkdir -p $@

# ------------------------------------------------------------------ SGEMM
$(BUILD)/test_sgemm: test/test_sgemm.cu $(LADDER) | $(BUILD)
	$(NVCC) $(FLAGS) $< -o $@ -lcublas

# `make test`  -> correctness suite
# `make bench` -> correctness + perf at 4096^3
# `make sweep` -> perf at several shapes + a summary table
# `make regs`  -> registers, spills and shared memory per kernel
.PHONY: test bench sweep regs
test: $(BUILD)/test_sgemm
	./$< --quick

bench: $(BUILD)/test_sgemm
	./$<

sweep: $(BUILD)/test_sgemm
	./$< --sweep

regs: test/test_sgemm.cu $(LADDER)
	$(NVCC) $(FLAGS) -Xptxas -v -c $< -o /dev/null

# ---------------------------------------------------------------- softmax
$(BUILD)/test_softmax: test/test_softmax.cu attention/softmax.cu | $(BUILD)
	$(NVCC) $(FLAGS) $< -o $@

.PHONY: softmax
softmax: $(BUILD)/test_softmax
	./$<

# -------------------------------------------------------------- attention
$(BUILD)/test_attn: test/test_attn.cu $(ATTN) | $(BUILD)
	$(NVCC) $(FLAGS) $< -o $@ -lcublas

# `make attn`       -> correctness suite (10 shapes)
# `make attn-bench` -> correctness + sweep over N
# `make attn-sweep` -> sweep only
.PHONY: attn attn-bench attn-sweep
attn: $(BUILD)/test_attn
	./$< --quick

attn-bench: $(BUILD)/test_attn
	./$<

attn-sweep: $(BUILD)/test_attn
	./$< --sweep

# staged debugging harness for the fused kernel (tile maps, row ratios)
$(BUILD)/test_online: test/test_online.cu attention/online_attn.cu | $(BUILD)
	$(NVCC) $(FLAGS) $< -o $@

.PHONY: online
online: $(BUILD)/test_online
	./$<

# ------------------------------------------------------------------ misc
# build every harness without running anything
.PHONY: all clean
all: $(BUILD)/test_sgemm $(BUILD)/test_softmax $(BUILD)/test_attn $(BUILD)/test_online

clean:
	rm -rf $(BUILD)
