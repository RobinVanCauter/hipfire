# Weight Read Reduction Plan for hipfire RDNA1

## Current State
- HFQ4-G256: 4.4GB model, 18 VGPRs, 282 GB/s effective, 59.8 tok/s
- Weight GEMV = 93% of forward time (~15.6ms of 16.7ms)
- Kernel ISA is near-optimal: v_bfe_u32 auto-emitted, scalar scale broadcast

## Option 1: E8 Lattice 2-bit (QuIP#-style)
- 4KB codebook in LDS, 256 entries × 8 fp16 values
- ds_read_b128 for codebook lookup (1 instruction per 8 weights)
- ~26 VGPRs → 100% occupancy maintained
- Model: 4.4GB → 2.2GB → projected ~120 tok/s
- FWHT preprocessing reuses existing kernel (0.09% overhead)
- Requires offline E8 lattice quantizer

## Option 2: Simple HFQ2 (no codebook)
- [f16 scale][f16 zero][2-bit × 256] = 68 bytes per 256 weights
- v_bfe_u32 width=2 (16 extractions per dword vs 8 for HFQ4)
- ~20 VGPRs → 100% occupancy
- Same compression as E8, simpler but needs FWHT for quality
- No codebook overhead, pure arithmetic dequant

## Option 3: Mixed 2-bit FFN + 4-bit Attention  
- FFN weights (75% of model) at 2-bit: 3.3GB → 1.65GB
- Attention weights (25%) at 4-bit: 1.1GB (unchanged)
- Total: 2.75GB → ~96 tok/s
- Quality: attention-critical layers preserved at higher precision

## ISA Key Findings (from disassembly)
- v_bfe_u32: auto-emitted by hipcc for all non-zero nibble positions
- v_cvt_f32_ubyte0: used for nibble→float (faster than fp16 intermediate)
- s_load_dwordx2: scalar broadcast for scale+zero (uses 16KB L1K cache)
- s_clause: compiler batches memory prefetches automatically
- ds_bpermute_b32: how __shfl_down/xor works on gfx1010 (via LDS)
- All kernels at 100% occupancy (first cliff at >52 VGPRs)

## Theoretical Ceiling
- 2-bit weights: 2.2GB per token
- At 282 GB/s effective: 2.2/0.282 = 7.8ms → 128 tok/s
- With EAGLE-3 (3.5 tokens per 9ms cycle): 389 tok/s theoretical!
