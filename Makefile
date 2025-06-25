# Makefile for Hash-Based Image Compression CUDA Project
# Based on research prompt requirements

# CUDA compiler
NVCC = nvcc

# Compiler flags
CFLAGS = -std=c++11 -O3
CUDA_FLAGS = -arch=sm_50 --ptxas-options=-v

# Libraries
LIBS = -lcurand

# Source files
HASH_DEMO_SRC = hash_demo.cu
HASH_ANALYSIS_SRC = hash_analysis.cu  
HASH_COMPRESSION_SRC = hash_compression.cu
ORIGINAL_MAIN_SRC = main.cu

# Executables
HASH_DEMO_EXE = hash_demo.exe
HASH_ANALYSIS_EXE = hash_analysis.exe
HASH_COMPRESSION_EXE = hash_compression.exe
ORIGINAL_MAIN_EXE = main.exe

# Default target
all: $(HASH_DEMO_EXE) $(HASH_ANALYSIS_EXE) $(HASH_COMPRESSION_EXE) $(ORIGINAL_MAIN_EXE)

# Build hash demonstration
$(HASH_DEMO_EXE): $(HASH_DEMO_SRC)
	$(NVCC) $(CFLAGS) $(CUDA_FLAGS) -o $@ $< $(LIBS)

# Build theoretical analysis
$(HASH_ANALYSIS_EXE): $(HASH_ANALYSIS_SRC)
	$(NVCC) $(CFLAGS) $(CUDA_FLAGS) -o $@ $< $(LIBS)

# Build full hash compression implementation
$(HASH_COMPRESSION_EXE): $(HASH_COMPRESSION_SRC)
	$(NVCC) $(CFLAGS) $(CUDA_FLAGS) -o $@ $< $(LIBS)

# Build original sum-based method for comparison
$(ORIGINAL_MAIN_EXE): $(ORIGINAL_MAIN_SRC)
	$(NVCC) $(CFLAGS) $(CUDA_FLAGS) -o $@ $< $(LIBS)

# Clean build files
clean:
	del /f *.exe *.obj *.lib *.exp 2>nul || true

# Run demonstrations
demo: $(HASH_DEMO_EXE)
	@echo Running hash compression demonstration...
	./$(HASH_DEMO_EXE)

analysis: $(HASH_ANALYSIS_EXE)
	@echo Running theoretical analysis...
	./$(HASH_ANALYSIS_EXE)

compression: $(HASH_COMPRESSION_EXE)
	@echo Running full hash compression test...
	./$(HASH_COMPRESSION_EXE)

original: $(ORIGINAL_MAIN_EXE)
	@echo Running original sum-based method...
	./$(ORIGINAL_MAIN_EXE)

# Compare methods
compare: $(HASH_DEMO_EXE) $(ORIGINAL_MAIN_EXE)
	@echo Comparing hash-based vs sum-based methods...
	@echo "=== Hash-Based Method ==="
	./$(HASH_DEMO_EXE)
	@echo ""
	@echo "=== Sum-Based Method ==="
	./$(ORIGINAL_MAIN_EXE)

# Run all tests
test: all
	@echo Running complete test suite...
	$(MAKE) analysis
	@echo ""
	$(MAKE) demo
	@echo ""
	$(MAKE) compare

# Help target
help:
	@echo Available targets:
	@echo   all         - Build all executables
	@echo   demo        - Run hash compression demonstration
	@echo   analysis    - Run theoretical analysis
	@echo   compression - Run full hash compression test
	@echo   original    - Run original sum-based method
	@echo   compare     - Compare both methods
	@echo   test        - Run complete test suite
	@echo   clean       - Remove build files
	@echo   help        - Show this help

.PHONY: all clean demo analysis compression original compare test help
