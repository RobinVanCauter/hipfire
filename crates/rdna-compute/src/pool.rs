//! GPU memory pool — eliminates hipMalloc/hipFree overhead in the hot loop.
//! Pre-allocates buffers of common sizes and reuses them via a free list.

use hip_bridge::{DeviceBuffer, HipResult, HipRuntime};
use std::collections::HashMap;

/// A pool of GPU buffers, bucketed by size.
/// Requesting a buffer returns one from the pool (if available) or allocates new.
/// Returning a buffer puts it back in the pool for reuse.
pub struct GpuPool {
    /// Free buffers bucketed by size (rounded up to power of 2)
    free_lists: HashMap<usize, Vec<DeviceBuffer>>,
    /// Total bytes currently allocated (for diagnostics)
    pub total_allocated: usize,
    pub total_reused: usize,
    pub total_new: usize,
}

impl GpuPool {
    pub fn new() -> Self {
        Self {
            free_lists: HashMap::new(),
            total_allocated: 0,
            total_reused: 0,
            total_new: 0,
        }
    }

    /// Round size up to next power of 2 (minimum 256 bytes) for bucketing.
    fn bucket_size(size: usize) -> usize {
        let min = 256;
        if size <= min {
            return min;
        }
        size.next_power_of_two()
    }

    /// Get a buffer of at least `size` bytes. Reuses from pool if available.
    pub fn alloc(&mut self, hip: &HipRuntime, size: usize) -> HipResult<DeviceBuffer> {
        let bucket = Self::bucket_size(size);
        if let Some(list) = self.free_lists.get_mut(&bucket) {
            if let Some(buf) = list.pop() {
                self.total_reused += 1;
                return Ok(buf);
            }
        }
        // No buffer available — allocate new
        self.total_new += 1;
        self.total_allocated += bucket;
        hip.malloc(bucket)
    }

    /// Return a buffer to the pool for reuse.
    pub fn free(&mut self, buf: DeviceBuffer) {
        let bucket = Self::bucket_size(buf.size());
        self.free_lists.entry(bucket).or_default().push(buf);
    }

    /// Actually free all pooled buffers (call on cleanup).
    pub fn drain(&mut self, hip: &HipRuntime) {
        for (_, list) in self.free_lists.drain() {
            for buf in list {
                let _ = hip.free(buf);
            }
        }
    }
}
