// ============================================================================
// ggml_vulkan_ext.cpp — Complete Vulkan Backend Extensions for OLLaMA/llama.cpp
// Target: Async upload, subgroup ops, pipelined scheduler, persistent buffers
// Compile: g++ -O3 -fPIC -shared -o libggml_vulkan_ext.so ggml_vulkan_ext.cpp -lvulkan
// ============================================================================

#include <vulkan/vulkan.h>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <algorithm>

#define CHECK_VK(cmd) do { VkResult r = cmd; if (r != VK_SUCCESS) { \
    fprintf(stderr, "Vulkan error %d at %s:%d\n", r, __FILE__, __LINE__); \
    return; } } while(0)

#define CHECK_VK_RET(cmd) do { VkResult r = cmd; if (r != VK_SUCCESS) { \
    fprintf(stderr, "Vulkan error %d at %s:%d\n", r, __FILE__, __LINE__); \
    return r; } } while(0)

// ============================================================================
// 1. ASYNC UPLOAD QUEUE — Background Transfer with Triple Buffering
// ============================================================================
// Uses a dedicated transfer queue (if available) or compute queue with
// async command buffer submission. Triple-buffered staging eliminates
// CPU wait on GPU upload completion.

struct VulkanAsyncUploader {
    VkDevice device;
    VkPhysicalDevice phys_device;
    VkQueue transfer_queue;
    uint32_t transfer_family;
    VkCommandPool cmd_pool;
    VkCommandBuffer cmd_bufs[3];  // triple buffer
    VkFence fences[3];
    VkDeviceSize staging_size;

    // Triple-buffered staging
    VkBuffer staging_bufs[3];
    VkDeviceMemory staging_mems[3];
    void* staging_ptrs[3];
    int current_buf;

    // Worker thread for background submission
    struct UploadJob {
        VkBuffer dst_buffer;
        VkDeviceSize size;
        int buf_idx;
    };

    std::thread worker;
    std::mutex mtx;
    std::condition_variable cv;
    std::queue<UploadJob> job_queue;
    std::atomic<bool> running;

    VulkanAsyncUploader(VkDevice dev, VkPhysicalDevice phys, VkDeviceSize buf_size)
        : device(dev), phys_device(phys), staging_size(buf_size), current_buf(0), running(true) {

        // Find transfer queue family (dedicated transfer preferred)
        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &queue_family_count, nullptr);
        std::vector<VkQueueFamilyProperties> queue_props(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &queue_family_count, queue_props.data());

        transfer_family = 0;
        bool found_dedicated = false;
        for (uint32_t i = 0; i < queue_family_count; i++) {
            if ((queue_props[i].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
                !(queue_props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
                !(queue_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
                transfer_family = i;
                found_dedicated = true;
                break;
            }
        }

        if (!found_dedicated) {
            // Fall back to compute queue family
            for (uint32_t i = 0; i < queue_family_count; i++) {
                if (queue_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    transfer_family = i;
                    break;
                }
            }
        }

        vkGetDeviceQueue(device, transfer_family, 0, &transfer_queue);

        // Command pool
        VkCommandPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = transfer_family;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        CHECK_VK(vkCreateCommandPool(device, &pool_info, nullptr, &cmd_pool));

        // Command buffers
        VkCommandBufferAllocateInfo cmd_alloc = {};
        cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmd_alloc.commandPool = cmd_pool;
        cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmd_alloc.commandBufferCount = 3;
        VkCommandBuffer bufs[3];
        CHECK_VK(vkAllocateCommandBuffers(device, &cmd_alloc, bufs));
        for (int i = 0; i < 3; i++) cmd_bufs[i] = bufs[i];

        // Fences
        for (int i = 0; i < 3; i++) {
            VkFenceCreateInfo fence_info = {};
            fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            CHECK_VK(vkCreateFence(device, &fence_info, nullptr, &fences[i]));
        }

        // Allocate staging buffers in host-visible, coherent memory
        VkPhysicalDeviceMemoryProperties mem_props;
        vkGetPhysicalDeviceMemoryProperties(phys, &mem_props);

        uint32_t host_visible_type = 0;
        for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
            if ((mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
                (mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                host_visible_type = i;
                break;
            }
        }

        for (int i = 0; i < 3; i++) {
            VkBufferCreateInfo buf_info = {};
            buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            buf_info.size = staging_size;
            buf_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            CHECK_VK(vkCreateBuffer(device, &buf_info, nullptr, &staging_bufs[i]));

            VkMemoryRequirements mem_req;
            vkGetBufferMemoryRequirements(device, staging_bufs[i], &mem_req);

            VkMemoryAllocateInfo alloc_info = {};
            alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            alloc_info.allocationSize = mem_req.size;
            alloc_info.memoryTypeIndex = host_visible_type;
            CHECK_VK(vkAllocateMemory(device, &alloc_info, nullptr, &staging_mems[i]));
            CHECK_VK(vkBindBufferMemory(device, staging_bufs[i], staging_mems[i], 0));
            CHECK_VK(vkMapMemory(device, staging_mems[i], 0, staging_size, 0, &staging_ptrs[i]));
        }

        // Start background worker thread
        worker = std::thread([this]() {
            while (running.load()) {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [this]() { return !job_queue.empty() || !running.load(); });

                while (!job_queue.empty()) {
                    UploadJob job = job_queue.front();
                    job_queue.pop();
                    lock.unlock();

                    // Wait for fence on this buffer index
                    vkWaitForFences(device, 1, &fences[job.buf_idx], VK_TRUE, UINT64_MAX);
                    vkResetFences(device, 1, &fences[job.buf_idx]);

                    // Record copy command
                    VkCommandBufferBeginInfo begin_info = {};
                    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                    vkBeginCommandBuffer(cmd_bufs[job.buf_idx], &begin_info);

                    VkBufferCopy copy_region = {};
                    copy_region.srcOffset = 0;
                    copy_region.dstOffset = 0;
                    copy_region.size = job.size;
                    vkCmdCopyBuffer(cmd_bufs[job.buf_idx], staging_bufs[job.buf_idx], job.dst_buffer, 1, &copy_region);

                    vkEndCommandBuffer(cmd_bufs[job.buf_idx]);

                    VkSubmitInfo submit_info = {};
                    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                    submit_info.commandBufferCount = 1;
                    submit_info.pCommandBuffers = &cmd_bufs[job.buf_idx];

                    vkQueueSubmit(transfer_queue, 1, &submit_info, fences[job.buf_idx]);

                    lock.lock();
                }
            }
        });
    }

    ~VulkanAsyncUploader() {
        running.store(false);
        cv.notify_all();
        if (worker.joinable()) worker.join();

        for (int i = 0; i < 3; i++) {
            vkUnmapMemory(device, staging_mems[i]);
            vkFreeMemory(device, staging_mems[i], nullptr);
            vkDestroyBuffer(device, staging_bufs[i], nullptr);
            vkDestroyFence(device, fences[i], nullptr);
        }
        vkFreeCommandBuffers(device, cmd_pool, 3, cmd_bufs);
        vkDestroyCommandPool(device, cmd_pool, nullptr);
    }

    // Returns host-visible pointer to fill with data
    void* begin_upload() {
        int idx = current_buf;
        // Check if fence is ready (non-blocking check)
        VkResult r = vkGetFenceStatus(device, fences[idx]);
        if (r == VK_NOT_READY) {
            // Wait for it
            vkWaitForFences(device, 1, &fences[idx], VK_TRUE, UINT64_MAX);
            vkResetFences(device, 1, &fences[idx]);
        } else if (r == VK_SUCCESS) {
            vkResetFences(device, 1, &fences[idx]);
        }
        return staging_ptrs[idx];
    }

    // Queues async upload. Returns immediately.
    void commit_upload(VkBuffer dst, VkDeviceSize size) {
        int idx = current_buf;
        UploadJob job;
        job.dst_buffer = dst;
        job.size = size;
        job.buf_idx = idx;

        {
            std::lock_guard<std::mutex> lock(mtx);
            job_queue.push(job);
        }
        cv.notify_one();

        current_buf = (current_buf + 1) % 3;
    }

    // Wait for all pending uploads
    void sync() {
        // Signal wait condition so worker drains the queue
        cv.notify_all();
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [this]() { return job_queue.empty(); });
        }
        vkQueueWaitIdle(transfer_queue);
    }
};

static VulkanAsyncUploader* g_vk_uploader = nullptr;

// ============================================================================
// 2. SUBGROUP OPS ENABLEMENT — Query + Shader Compilation Hints
// ============================================================================
// Queries device subgroup capabilities and sets up optimal shader paths.

struct VulkanSubgroupCaps {
    VkPhysicalDeviceSubgroupProperties props;
    bool supported;
    uint32_t size;
    VkSubgroupFeatureFlags features;
};

static VulkanSubgroupCaps g_subgroup_caps = {};

extern "C" {

void vulkan_query_subgroup_caps(VkPhysicalDevice phys) {
    VkPhysicalDeviceSubgroupProperties subgroup_props = {};
    subgroup_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;

    VkPhysicalDeviceProperties2 props2 = {};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &subgroup_props;

    vkGetPhysicalDeviceProperties2(phys, &props2);

    g_subgroup_caps.props = subgroup_props;
    g_subgroup_caps.supported = true;
    g_subgroup_caps.size = subgroup_props.subgroupSize;
    g_subgroup_caps.features = subgroup_props.supportedOperations;

    fprintf(stderr, "[VulkanExt] Subgroup size: %d, supported ops: 0x%x\n",
            subgroup_props.subgroupSize, subgroup_props.supportedOperations);
    fprintf(stderr, "[VulkanExt] Quad: %s, Arithmetic: %s, Ballot: %s, Shuffle: %s\n",
            (subgroup_props.supportedOperations & VK_SUBGROUP_FEATURE_QUAD_BIT) ? "yes" : "no",
            (subgroup_props.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) ? "yes" : "no",
            (subgroup_props.supportedOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT) ? "yes" : "no",
            (subgroup_props.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT) ? "yes" : "no");
}

// Returns optimal subgroup size for shader compilation (32 for AMD RDNA)
int vulkan_get_optimal_subgroup_size() {
    if (g_subgroup_caps.supported) {
        // AMD RDNA prefers wave32 (32) for compute
        if (g_subgroup_caps.size == 64) return 32; // Use wave32 mode if available
        return (int)g_subgroup_caps.size;
    }
    return 32; // safe default
}

// Sets environment hints for external shader compilers (glslang, slang)
void vulkan_set_subgroup_hints() {
    char buf[32];
    snprintf(buf, sizeof(buf), "%d", vulkan_get_optimal_subgroup_size());
#ifdef _WIN32
    _putenv_s("VK_SUBGROUP_SIZE", buf);
    _putenv_s("VK_SPIRV_SUBGROUP_OPS", "1");
    if (g_subgroup_caps.features & VK_SUBGROUP_FEATURE_BALLOT_BIT) {
        _putenv_s("VK_USE_SUBGROUP_BALLOT", "1");
    }
    if (g_subgroup_caps.features & VK_SUBGROUP_FEATURE_SHUFFLE_BIT) {
        _putenv_s("VK_USE_SUBGROUP_SHUFFLE", "1");
    }
    if (g_subgroup_caps.features & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) {
        _putenv_s("VK_USE_SUBGROUP_ARITHMETIC", "1");
    }
#else
    setenv("VK_SUBGROUP_SIZE", buf, 1);
    setenv("VK_SPIRV_SUBGROUP_OPS", "1", 1);
    if (g_subgroup_caps.features & VK_SUBGROUP_FEATURE_BALLOT_BIT) {
        setenv("VK_USE_SUBGROUP_BALLOT", "1", 1);
    }
    if (g_subgroup_caps.features & VK_SUBGROUP_FEATURE_SHUFFLE_BIT) {
        setenv("VK_USE_SUBGROUP_SHUFFLE", "1", 1);
    }
    if (g_subgroup_caps.features & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) {
        setenv("VK_USE_SUBGROUP_ARITHMETIC", "1", 1);
    }
#endif
}

} // extern "C"

// ============================================================================
// 3. PIPELINED SCHEDULER — Triple-Buffered Command Submission
// ============================================================================
// Eliminates CPU bubbles by keeping 3 frames in flight. The GPU never stalls
// waiting for the CPU to record commands.

struct VulkanPipelinedScheduler {
    static constexpr int NUM_FRAMES = 3;

    VkDevice device;
    VkQueue compute_queue;
    uint32_t compute_family;

    VkCommandPool cmd_pools[NUM_FRAMES];
    VkCommandBuffer cmd_bufs[NUM_FRAMES];
    VkFence fences[NUM_FRAMES];
    VkSemaphore semaphores[NUM_FRAMES]; // for cross-queue sync if needed
    int current_frame;

    // Stats
    std::atomic<uint64_t> frames_submitted;
    std::atomic<uint64_t> frames_completed;

    VulkanPipelinedScheduler(VkDevice dev, uint32_t queue_family)
        : device(dev), compute_family(queue_family), current_frame(0),
          frames_submitted(0), frames_completed(0) {

        vkGetDeviceQueue(device, queue_family, 0, &compute_queue);

        for (int i = 0; i < NUM_FRAMES; i++) {
            VkCommandPoolCreateInfo pool_info = {};
            pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            pool_info.queueFamilyIndex = queue_family;
            pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            CHECK_VK(vkCreateCommandPool(device, &pool_info, nullptr, &cmd_pools[i]));

            VkCommandBufferAllocateInfo cmd_info = {};
            cmd_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cmd_info.commandPool = cmd_pools[i];
            cmd_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cmd_info.commandBufferCount = 1;
            CHECK_VK(vkAllocateCommandBuffers(device, &cmd_info, &cmd_bufs[i]));

            VkFenceCreateInfo fence_info = {};
            fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            CHECK_VK(vkCreateFence(device, &fence_info, nullptr, &fences[i]));

            VkSemaphoreCreateInfo sem_info = {};
            sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            CHECK_VK(vkCreateSemaphore(device, &sem_info, nullptr, &semaphores[i]));
        }
    }

    ~VulkanPipelinedScheduler() {
        // Wait for all in-flight
        vkQueueWaitIdle(compute_queue);
        for (int i = 0; i < NUM_FRAMES; i++) {
            vkDestroySemaphore(device, semaphores[i], nullptr);
            vkDestroyFence(device, fences[i], nullptr);
            vkFreeCommandBuffers(device, cmd_pools[i], 1, &cmd_bufs[i]);
            vkDestroyCommandPool(device, cmd_pools[i], nullptr);
        }
    }

    // Acquire a command buffer for recording. Blocks if all 3 frames in flight.
    VkCommandBuffer acquire_frame() {
        int frame = current_frame;

        // Wait for this frame's fence if it's still in flight
        VkResult r = vkGetFenceStatus(device, fences[frame]);
        if (r == VK_NOT_READY) {
            vkWaitForFences(device, 1, &fences[frame], VK_TRUE, UINT64_MAX);
        }
        vkResetFences(device, 1, &fences[frame]);

        // Reset and begin recording
        vkResetCommandBuffer(cmd_bufs[frame], 0);

        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd_bufs[frame], &begin_info);

        return cmd_bufs[frame];
    }

    // Submit the current frame and advance
    void submit_frame() {
        int frame = current_frame;
        vkEndCommandBuffer(cmd_bufs[frame]);

        VkSubmitInfo submit_info = {};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd_bufs[frame];

        // Signal semaphore for next frame (optional chaining)
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = &semaphores[frame];

        vkQueueSubmit(compute_queue, 1, &submit_info, fences[frame]);
        frames_submitted.fetch_add(1);

        current_frame = (current_frame + 1) % NUM_FRAMES;
    }

    // Wait for all frames to complete (drain pipeline)
    void drain() {
        vkQueueWaitIdle(compute_queue);
        for (int i = 0; i < NUM_FRAMES; i++) {
            vkResetFences(device, 1, &fences[i]);
        }
    }

    // Check if GPU is idle (all frames done)
    bool is_idle() {
        for (int i = 0; i < NUM_FRAMES; i++) {
            if (vkGetFenceStatus(device, fences[i]) == VK_NOT_READY) {
                return false;
            }
        }
        return true;
    }
};

static VulkanPipelinedScheduler* g_vk_scheduler = nullptr;

// ============================================================================
// 4. PERSISTENT DESCRIPTOR SETS — Eliminate per-kernel descriptor allocation
// ============================================================================
// Pre-allocates descriptor sets for common tensor operations to avoid
// vkAllocateDescriptorSets overhead.

struct VulkanPersistentDescriptors {
    VkDevice device;
    VkDescriptorPool pool;
    VkDescriptorSetLayout layout;
    std::vector<VkDescriptorSet> sets;
    int current_set;

    static constexpr int MAX_SETS = 64;
    static constexpr int BINDINGS_PER_SET = 4; // srcA, srcB, dst, params

    VulkanPersistentDescriptors(VkDevice dev) : device(dev), current_set(0) {
        // Create layout
        VkDescriptorSetLayoutBinding bindings[4] = {};
        for (int i = 0; i < 4; i++) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo layout_info = {};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = 4;
        layout_info.pBindings = bindings;
        vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &layout);

        // Create pool
        VkDescriptorPoolSize pool_size = {};
        pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = MAX_SETS * BINDINGS_PER_SET;

        VkDescriptorPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.maxSets = MAX_SETS;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        vkCreateDescriptorPool(device, &pool_info, nullptr, &pool);

        // Allocate all sets upfront
        sets.resize(MAX_SETS);
        std::vector<VkDescriptorSetLayout> layouts(MAX_SETS, layout);
        VkDescriptorSetAllocateInfo alloc_info = {};
        alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool = pool;
        alloc_info.descriptorSetCount = MAX_SETS;
        alloc_info.pSetLayouts = layouts.data();
        vkAllocateDescriptorSets(device, &alloc_info, sets.data());
    }

    ~VulkanPersistentDescriptors() {
        vkDestroyDescriptorPool(device, pool, nullptr);
        vkDestroyDescriptorSetLayout(device, layout, nullptr);
    }

    VkDescriptorSet acquire_set() {
        int idx = current_set;
        current_set = (current_set + 1) % MAX_SETS;
        return sets[idx];
    }

    void update_buffer_binding(VkDescriptorSet set, int binding, VkBuffer buffer, VkDeviceSize size) {
        VkDescriptorBufferInfo buf_info = {};
        buf_info.buffer = buffer;
        buf_info.offset = 0;
        buf_info.range = size;

        VkWriteDescriptorSet write = {};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = set;
        write.dstBinding = binding;
        write.dstArrayElement = 0;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.descriptorCount = 1;
        write.pBufferInfo = &buf_info;

        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }
};

static VulkanPersistentDescriptors* g_vk_descriptors = nullptr;

// ============================================================================
// 5. MEMORY ALLOCATION POOL — Fast device memory for tensors
// ============================================================================
// Sub-allocates from large device allocations to avoid vkAllocateMemory overhead.

struct VulkanMemoryPool {
    VkDevice device;
    VkPhysicalDevice phys_device;
    VkDeviceMemory pool_mem;
    VkDeviceSize pool_size;
    VkDeviceSize used;
    void* mapped_ptr;
    bool host_visible;

    VulkanMemoryPool(VkDevice dev, VkPhysicalDevice phys, VkDeviceSize size, bool host_vis)
        : device(dev), phys_device(phys), pool_size(size), used(0), host_visible(host_vis), mapped_ptr(nullptr) {

        VkPhysicalDeviceMemoryProperties mem_props;
        vkGetPhysicalDeviceMemoryProperties(phys, &mem_props);

        uint32_t mem_type = 0;
        VkMemoryPropertyFlags wanted = host_vis 
            ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
            : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
            if ((mem_props.memoryTypes[i].propertyFlags & wanted) == wanted) {
                mem_type = i;
                break;
            }
        }

        VkMemoryAllocateInfo alloc_info = {};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = size;
        alloc_info.memoryTypeIndex = mem_type;
        vkAllocateMemory(device, &alloc_info, nullptr, &pool_mem);

        if (host_vis) {
            vkMapMemory(device, pool_mem, 0, size, 0, &mapped_ptr);
        }
    }

    ~VulkanMemoryPool() {
        if (mapped_ptr) vkUnmapMemory(device, pool_mem);
        vkFreeMemory(device, pool_mem, nullptr);
    }

    // Simple bump allocator
    VkDeviceSize alloc(VkDeviceSize size, VkDeviceSize alignment) {
        VkDeviceSize aligned = (used + alignment - 1) & ~(alignment - 1);
        if (aligned + size > pool_size) return VK_WHOLE_SIZE; // fail
        VkDeviceSize offset = aligned;
        used = aligned + size;
        return offset;
    }

    void reset() {
        used = 0;
    }
};

static VulkanMemoryPool* g_vk_device_pool = nullptr;
static VulkanMemoryPool* g_vk_staging_pool = nullptr;

// ============================================================================
// C API — Extern "C" Interface
// ============================================================================

extern "C" {

// --- Async Upload ---
void vulkan_async_upload_init(VkDevice device, VkPhysicalDevice phys, uint64_t max_size) {
    if (g_vk_uploader) delete g_vk_uploader;
    g_vk_uploader = new VulkanAsyncUploader(device, phys, max_size);
}

void* vulkan_async_upload_begin() {
    if (!g_vk_uploader) return nullptr;
    return g_vk_uploader->begin_upload();
}

void vulkan_async_upload_commit(VkBuffer dst, uint64_t size) {
    if (g_vk_uploader) g_vk_uploader->commit_upload(dst, size);
}

void vulkan_async_upload_sync() {
    if (g_vk_uploader) g_vk_uploader->sync();
}

// --- Subgroup Ops ---
void vulkan_subgroup_init(VkPhysicalDevice phys) {
    vulkan_query_subgroup_caps(phys);
    vulkan_set_subgroup_hints();
}

int vulkan_subgroup_get_size() {
    return vulkan_get_optimal_subgroup_size();
}

// --- Pipelined Scheduler ---
void vulkan_scheduler_init(VkDevice device, uint32_t queue_family) {
    if (g_vk_scheduler) delete g_vk_scheduler;
    g_vk_scheduler = new VulkanPipelinedScheduler(device, queue_family);
}

VkCommandBuffer vulkan_scheduler_acquire() {
    if (!g_vk_scheduler) return VK_NULL_HANDLE;
    return g_vk_scheduler->acquire_frame();
}

void vulkan_scheduler_submit() {
    if (g_vk_scheduler) g_vk_scheduler->submit_frame();
}

void vulkan_scheduler_drain() {
    if (g_vk_scheduler) g_vk_scheduler->drain();
}

int vulkan_scheduler_is_idle() {
    if (!g_vk_scheduler) return 1;
    return g_vk_scheduler->is_idle() ? 1 : 0;
}

// --- Persistent Descriptors ---
void vulkan_descriptor_init(VkDevice device) {
    if (g_vk_descriptors) delete g_vk_descriptors;
    g_vk_descriptors = new VulkanPersistentDescriptors(device);
}

VkDescriptorSet vulkan_descriptor_acquire() {
    if (!g_vk_descriptors) return VK_NULL_HANDLE;
    return g_vk_descriptors->acquire_set();
}

void vulkan_descriptor_update_buffer(VkDescriptorSet set, int binding, VkBuffer buffer, uint64_t size) {
    if (g_vk_descriptors) g_vk_descriptors->update_buffer_binding(set, binding, buffer, size);
}

// --- Memory Pools ---
void vulkan_memory_pool_init(VkDevice device, VkPhysicalDevice phys, uint64_t device_size, uint64_t staging_size) {
    if (g_vk_device_pool) delete g_vk_device_pool;
    if (g_vk_staging_pool) delete g_vk_staging_pool;

    g_vk_device_pool = new VulkanMemoryPool(device, phys, device_size, false);
    g_vk_staging_pool = new VulkanMemoryPool(device, phys, staging_size, true);
}

uint64_t vulkan_memory_pool_alloc_device(uint64_t size, uint64_t alignment) {
    if (!g_vk_device_pool) return VK_WHOLE_SIZE;
    return g_vk_device_pool->alloc(size, alignment);
}

uint64_t vulkan_memory_pool_alloc_staging(uint64_t size, uint64_t alignment) {
    if (!g_vk_staging_pool) return VK_WHOLE_SIZE;
    return g_vk_staging_pool->alloc(size, alignment);
}

void vulkan_memory_pool_reset() {
    if (g_vk_device_pool) g_vk_device_pool->reset();
    if (g_vk_staging_pool) g_vk_staging_pool->reset();
}

VkDeviceMemory vulkan_memory_pool_get_device_mem() {
    if (!g_vk_device_pool) return VK_NULL_HANDLE;
    return g_vk_device_pool->pool_mem;
}

VkDeviceMemory vulkan_memory_pool_get_staging_mem() {
    if (!g_vk_staging_pool) return VK_NULL_HANDLE;
    return g_vk_staging_pool->pool_mem;
}

void* vulkan_memory_pool_get_staging_ptr() {
    if (!g_vk_staging_pool) return nullptr;
    return g_vk_staging_pool->mapped_ptr;
}

} // extern "C"
