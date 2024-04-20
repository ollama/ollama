#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include "ggml.h"
#include "ggml-impl.h"
#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MAX_FREE_BLOCKS 256

//#define GGML_ALLOCATOR_DEBUG

//#define AT_PRINTF(...) fprintf(stderr, __VA_ARGS__)
#define AT_PRINTF(...)


static bool ggml_is_view(const struct ggml_tensor * t) {
    return t->view_src != NULL;
}

static bool ggml_are_same_layout(const struct ggml_tensor * a, const struct ggml_tensor * b) {
    if (a->type != b->type) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (a->ne[i] != b->ne[i]) {
            return false;
        }
        if (a->nb[i] != b->nb[i]) {
            return false;
        }
    }
    return true;
}

static bool ggml_op_can_inplace(enum ggml_op op) {
    switch (op) {
        case GGML_OP_SCALE:
        case GGML_OP_DIAG_MASK_ZERO:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_ADD:
        case GGML_OP_ADD1:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_UNARY:
        case GGML_OP_ROPE:
        case GGML_OP_RMS_NORM:
        case GGML_OP_SOFT_MAX:
            return true;

        default:
            return false;
    }
}

static size_t aligned_offset(const void * buffer, size_t offset, size_t alignment) {
    assert(alignment && !(alignment & (alignment - 1))); // power of 2
    size_t align = (alignment - (((uintptr_t)buffer + offset) % alignment)) % alignment;
    return offset + align;
}

// tallocr

struct ggml_tallocr ggml_tallocr_new(ggml_backend_buffer_t buffer) {
    void * base = ggml_backend_buffer_get_base(buffer);
    size_t align = ggml_backend_buffer_get_alignment(buffer);

    assert(align && !(align & (align - 1))); // power of 2

    struct ggml_tallocr talloc = (struct ggml_tallocr) {
        /*.buffer    = */ buffer,
        /*.base      = */ base,
        /*.alignment = */ align,
        /*.offset    = */ aligned_offset(base, 0, align),
    };
    return talloc;
}

void ggml_tallocr_alloc(struct ggml_tallocr * talloc, struct ggml_tensor * tensor) {
    size_t size = ggml_backend_buffer_get_alloc_size(talloc->buffer, tensor);
    size = GGML_PAD(size, talloc->alignment);

    if (talloc->offset + size > ggml_backend_buffer_get_size(talloc->buffer)) {
        fprintf(stderr, "%s: not enough space in the buffer to allocate %s (needed %zu, available %zu)\n",
                __func__, tensor->name, size, ggml_backend_buffer_get_size(talloc->buffer) - talloc->offset);
        GGML_ASSERT(!"not enough space in the buffer");
        return;
    }

    void * addr = (char *)ggml_backend_buffer_get_base(talloc->buffer) + talloc->offset;
    talloc->offset += size;

    assert(((uintptr_t)addr % talloc->alignment) == 0);

    ggml_backend_tensor_alloc(talloc->buffer, tensor, addr);
}

// dynamic tensor allocator

struct free_block {
    size_t offset;
    size_t size;
};

struct ggml_dyn_tallocr {
    size_t alignment;
    int n_free_blocks;
    struct free_block free_blocks[MAX_FREE_BLOCKS];
    size_t max_size;

#ifdef GGML_ALLOCATOR_DEBUG
    struct {
        const struct ggml_tensor * tensor;
        size_t offset;
    } allocated_tensors[1024];
#endif
};

#ifdef GGML_ALLOCATOR_DEBUG
static void add_allocated_tensor(struct ggml_dyn_tallocr * alloc, size_t offset, const struct ggml_tensor * tensor) {
    for (int i = 0; i < 1024; i++) {
        if (alloc->allocated_tensors[i].tensor == NULL) {
            alloc->allocated_tensors[i].tensor = tensor;
            alloc->allocated_tensors[i].offset = offset;
            return;
        }
    }
    GGML_ASSERT(!"out of allocated_tensors");
}
static void remove_allocated_tensor(struct ggml_dyn_tallocr * alloc, size_t offset, const struct ggml_tensor * tensor) {
    for (int i = 0; i < 1024; i++) {
        if (alloc->allocated_tensors[i].offset == offset) {
            alloc->allocated_tensors[i].tensor = NULL;
            return;
        }
    }
    fprintf(stderr, "tried to free tensor %s not found\n", tensor->name);
    GGML_ASSERT(!"tensor not found");
}
#endif

static size_t ggml_dyn_tallocr_alloc(struct ggml_dyn_tallocr * alloc, size_t size, const struct ggml_tensor * tensor) {
    size = aligned_offset(NULL, size, alloc->alignment);

    AT_PRINTF("%s: allocating %s (%zu bytes) - ", __func__, tensor->name, size);

    size_t max_avail = 0;

    // find the best fitting free block besides the last block
    int best_fit_block = -1;
    size_t best_fit_size = SIZE_MAX;
    for (int i = 0; i < alloc->n_free_blocks - 1; i++) {
        struct free_block * block = &alloc->free_blocks[i];
        max_avail = MAX(max_avail, block->size);
        if (block->size >= size && block->size <= best_fit_size) {
            best_fit_block = i;
            best_fit_size = block->size;
        }
    }

    if (best_fit_block == -1) {
        // the last block is our last resort
        struct free_block * block = &alloc->free_blocks[alloc->n_free_blocks - 1];
        max_avail = MAX(max_avail, block->size);
        if (block->size >= size) {
            best_fit_block = alloc->n_free_blocks - 1;
        } else {
            // this should never happen
            fprintf(stderr, "%s: not enough space in the buffer to allocate %zu bytes, largest block available %zu bytes\n",
                    __func__, size, max_avail);
            GGML_ASSERT(!"not enough space in the buffer");
            GGML_UNREACHABLE();
        }
    }

    struct free_block * block = &alloc->free_blocks[best_fit_block];
    size_t offset = block->offset;
    block->offset = offset + size;
    block->size -= size;
    if (block->size == 0) {
        // remove block if empty
        alloc->n_free_blocks--;
        for (int j = best_fit_block; j < alloc->n_free_blocks; j++) {
            alloc->free_blocks[j] = alloc->free_blocks[j+1];
        }
    }

    AT_PRINTF("block %d, offset %zu\n", best_fit_block, offset);

#ifdef GGML_ALLOCATOR_DEBUG
    add_allocated_tensor(alloc, offset, tensor);
    size_t cur_max = offset + size;
    if (cur_max > alloc->max_size) {
        // sort allocated_tensors by offset
        for (int i = 0; i < 1024; i++) {
            for (int j = i + 1; j < 1024; j++) {
                if (alloc->allocated_tensors[i].offset > alloc->allocated_tensors[j].offset) {
                    const struct ggml_tensor * tmp_tensor = alloc->allocated_tensors[i].tensor;
                    size_t tmp_offset = alloc->allocated_tensors[i].offset;
                    alloc->allocated_tensors[i].tensor = alloc->allocated_tensors[j].tensor;
                    alloc->allocated_tensors[i].offset = alloc->allocated_tensors[j].offset;
                    alloc->allocated_tensors[j].tensor = tmp_tensor;
                    alloc->allocated_tensors[j].offset = tmp_offset;
                }
            }
        }
        fprintf(stderr, "max_size = %.2f MB: tensors: ", cur_max / 1024.0 / 1024.0);
        for (int i = 0; i < 1024; i++) {
            if (alloc->allocated_tensors[i].tensor) {
                fprintf(stderr, "%s [%zx-%zx] (%.2f MB) ", alloc->allocated_tensors[i].tensor->name,
                    alloc->allocated_tensors[i].offset,
                    alloc->allocated_tensors[i].offset + ggml_nbytes(alloc->allocated_tensors[i].tensor),
                    ggml_nbytes(alloc->allocated_tensors[i].tensor) / 1024.0 / 1024.0);
            }
        }
        fprintf(stderr, "\n");
    }
#endif

    alloc->max_size = MAX(alloc->max_size, offset + size);

    return offset;

    GGML_UNUSED(tensor);
}

// this is a very naive implementation, but for our case the number of free blocks should be very small
static void ggml_dyn_tallocr_free_tensor(struct ggml_dyn_tallocr * alloc, size_t offset, size_t size, const struct ggml_tensor * tensor) {
    size = aligned_offset(NULL, size, alloc->alignment);

    AT_PRINTF("%s: freeing %s at %zu (%zu bytes) - n_free_blocks = %d\n", __func__, tensor->name, offset, size, alloc->n_free_blocks);

#ifdef GGML_ALLOCATOR_DEBUG
    remove_allocated_tensor(alloc, offset, tensor);
#endif

    // see if we can merge with an existing block
    for (int i = 0; i < alloc->n_free_blocks; i++) {
        struct free_block * block = &alloc->free_blocks[i];
        // check if ptr is at the end of the block
        if (block->offset + block->size == offset) {
            block->size += size;
            // check if we can merge with the next block
            if (i < alloc->n_free_blocks - 1 && block->offset + block->size == alloc->free_blocks[i+1].offset) {
                block->size += alloc->free_blocks[i+1].size;
                alloc->n_free_blocks--;
                for (int j = i+1; j < alloc->n_free_blocks; j++) {
                    alloc->free_blocks[j] = alloc->free_blocks[j+1];
                }
            }
            return;
        }
        // check if ptr is at the beginning of the block
        if (offset + size == block->offset) {
            block->offset = offset;
            block->size += size;
            // check if we can merge with the previous block
            if (i > 0 && alloc->free_blocks[i-1].offset + alloc->free_blocks[i-1].size == block->offset) {
                alloc->free_blocks[i-1].size += block->size;
                alloc->n_free_blocks--;
                for (int j = i; j < alloc->n_free_blocks; j++) {
                    alloc->free_blocks[j] = alloc->free_blocks[j+1];
                }
            }
            return;
        }
    }
    // otherwise, add a new block
    GGML_ASSERT(alloc->n_free_blocks < MAX_FREE_BLOCKS && "out of free blocks");
    // insert the new block in the correct position to keep the array sorted by address (to make merging blocks faster)
    int insert_pos = 0;
    while (insert_pos < alloc->n_free_blocks && alloc->free_blocks[insert_pos].offset < offset) {
        insert_pos++;
    }
    // shift all blocks from insert_pos onward to make room for the new block
    for (int i = alloc->n_free_blocks; i > insert_pos; i--) {
        alloc->free_blocks[i] = alloc->free_blocks[i-1];
    }
    // insert the new block
    alloc->free_blocks[insert_pos].offset = offset;
    alloc->free_blocks[insert_pos].size = size;
    alloc->n_free_blocks++;

    GGML_UNUSED(tensor);
}

static void ggml_dyn_tallocr_reset(struct ggml_dyn_tallocr * alloc) {
    alloc->n_free_blocks = 1;
    alloc->free_blocks[0].offset = 0;
    alloc->free_blocks[0].size = SIZE_MAX/2; // restrict maximum size of a measure allocator to half size_t max to avoid overflows
    alloc->max_size = 0;
}

static struct ggml_dyn_tallocr * ggml_dyn_tallocr_new(size_t alignment) {
    struct ggml_dyn_tallocr * alloc = (struct ggml_dyn_tallocr *)malloc(sizeof(struct ggml_dyn_tallocr));

    *alloc = (struct ggml_dyn_tallocr) {
        /*.alignment     = */ alignment,
        /*.n_free_blocks = */ 0,
        /*.free_blocks   = */ {{0}},
        /*.max_size      = */ 0,
#ifdef GGML_ALLOCATOR_DEBUG
        /*.allocated_tensors = */ {{0}},
#endif
    };

    ggml_dyn_tallocr_reset(alloc);

    return alloc;
}

static void ggml_dyn_tallocr_free(struct ggml_dyn_tallocr * alloc) {
    free(alloc);
}

static size_t ggml_dyn_tallocr_max_size(struct ggml_dyn_tallocr * alloc) {
    return alloc->max_size;
}


/////////////////////////////////////

// graph allocator

struct hash_node {
    int n_children;
    int n_views;
    int buffer_id;
    size_t offset; // offset within the buffer
    bool allocated;
};

struct tensor_alloc {
    size_t offset;
    size_t size_max; // 0 = pre-allocated, unused, or view
};

struct leaf_alloc {
    int buffer_id;
    struct tensor_alloc leaf;
};

struct node_alloc {
    int buffer_id;
    struct tensor_alloc dst;
    struct tensor_alloc src[GGML_MAX_SRC];
};

struct ggml_gallocr {
    ggml_backend_buffer_type_t * bufts; // [n_buffers]
    ggml_backend_buffer_t * buffers; // [n_buffers]
    struct ggml_dyn_tallocr ** buf_tallocs; // [n_buffers]
    int n_buffers;

    struct ggml_hash_set hash_set;
    struct hash_node * hash_values; // [hash_set.size]

    struct node_alloc * node_allocs; // [n_nodes]
    int n_nodes;

    struct leaf_alloc * leaf_allocs; // [n_leafs]
    int n_leafs;
};

ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_buffer_type_t * bufts, int n_bufs) {
    ggml_gallocr_t galloc = (ggml_gallocr_t)calloc(sizeof(struct ggml_gallocr), 1);
    GGML_ASSERT(galloc != NULL);

    galloc->bufts = calloc(sizeof(ggml_backend_buffer_type_t) * n_bufs, 1);
    GGML_ASSERT(galloc->bufts != NULL);

    galloc->buffers = calloc(sizeof(ggml_backend_buffer_t) * n_bufs, 1);
    GGML_ASSERT(galloc->buffers != NULL);

    galloc->buf_tallocs = calloc(sizeof(struct ggml_dyn_tallocr *) * n_bufs, 1);
    GGML_ASSERT(galloc->buf_tallocs != NULL);

    for (int i = 0; i < n_bufs; i++) {
        galloc->bufts[i] = bufts[i];
        galloc->buffers[i] = NULL;
        size_t alignment = ggml_backend_buft_get_alignment(bufts[i]);
        galloc->buf_tallocs[i] = ggml_dyn_tallocr_new(alignment);
    }
    galloc->n_buffers = n_bufs;

    return galloc;
}

ggml_gallocr_t ggml_gallocr_new(ggml_backend_buffer_type_t buft) {
    return ggml_gallocr_new_n(&buft, 1);
}

void ggml_gallocr_free(ggml_gallocr_t galloc) {
    if (galloc == NULL) {
        return;
    }

    for (int i = 0; i < galloc->n_buffers; i++) {
        if (galloc->buffers != NULL) {
            ggml_backend_buffer_free(galloc->buffers[i]);
        }
        if (galloc->buf_tallocs != NULL) {
            ggml_dyn_tallocr_free(galloc->buf_tallocs[i]);
        }
    }

    free(galloc->hash_set.keys);
    free(galloc->hash_values);
    free(galloc->bufts);
    free(galloc->buffers);
    free(galloc->buf_tallocs);
    free(galloc->node_allocs);
    free(galloc->leaf_allocs);
    free(galloc);
}

typedef struct ggml_gallocr * ggml_gallocr_t;

static struct hash_node * ggml_gallocr_hash_get(ggml_gallocr_t galloc, struct ggml_tensor * t) {
    size_t i = ggml_hash_find_or_insert(galloc->hash_set, t);
    return &galloc->hash_values[i];
}

static bool ggml_gallocr_is_own(ggml_gallocr_t galloc, struct ggml_tensor * t) {
    return ggml_gallocr_hash_get(galloc, t)->allocated;
}

static void ggml_gallocr_set_node_offset(ggml_gallocr_t galloc, struct ggml_tensor * node, int buffer_id, size_t offset) {
    struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);
    hn->buffer_id = buffer_id;
    hn->offset = offset;
    hn->allocated = true;
}

static bool ggml_gallocr_is_allocated(ggml_gallocr_t galloc, struct ggml_tensor * t) {
    return t->data != NULL || ggml_gallocr_hash_get(galloc, t)->allocated;
}

static void ggml_gallocr_allocate_node(ggml_gallocr_t galloc, struct ggml_tensor * node, int buffer_id) {
    struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);

    if (!ggml_gallocr_is_allocated(galloc, node) && !ggml_is_view(node)) {
        hn->allocated = true;
        assert(hn->offset == 0);

        // try to reuse a parent's buffer (inplace)
        if (ggml_op_can_inplace(node->op)) {
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                struct ggml_tensor * parent = node->src[i];
                if (parent == NULL) {
                    continue;
                }

                // if the node's data is external, then we cannot re-use it
                if (!ggml_gallocr_is_own(galloc, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as %p is external\n", parent->name, node->name, parent->data);
                    continue;
                }

                // outputs cannot be reused
                if (parent->flags & GGML_TENSOR_FLAG_OUTPUT || (parent->view_src != NULL && parent->view_src->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                    AT_PRINTF("not reusing parent %s for %s as it is an output\n", parent->name, node->name);
                    continue;
                }

                if (!ggml_are_same_layout(node, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as layouts are different\n", parent->name, node->name);
                    continue;
                }

                struct hash_node * p_hn = ggml_gallocr_hash_get(galloc, parent);
                if (p_hn->n_children == 1 && p_hn->n_views == 0) {
                    if (ggml_is_view(parent)) {
                        struct ggml_tensor * view_src = parent->view_src;
                        struct hash_node * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                        if (view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && view_src->data == parent->data) {
                            AT_PRINTF("reusing view parent %s (%s) for %s\n", parent->name, view_src->name, node->name);
                            assert(view_src_hn->offset == p_hn->offset);
                            hn->buffer_id = p_hn->buffer_id;
                            hn->offset = p_hn->offset;
                            p_hn->allocated = false; // avoid freeing the parent
                            view_src_hn->allocated = false;
                            return;
                        }
                    } else {
                        AT_PRINTF("reusing parent %s for %s\n", parent->name, node->name);
                        hn->buffer_id = p_hn->buffer_id;
                        hn->offset = p_hn->offset;
                        p_hn->allocated = false; // avoid freeing the parent
                        return;
                    }
                }
            }
        }
        // allocate tensor from the buffer
        struct ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
        ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
        size_t size = ggml_backend_buft_get_alloc_size(buft, node);
        size_t offset = ggml_dyn_tallocr_alloc(alloc, size, node);
        hn->buffer_id = buffer_id;
        hn->offset = offset;
        return;
    }
}

static void ggml_gallocr_free_node(ggml_gallocr_t galloc, struct ggml_tensor * node, int buffer_id) {
    // graph outputs are never freed
    if (node->flags & GGML_TENSOR_FLAG_OUTPUT) {
        AT_PRINTF("not freeing output %s\n", node->name);
        return;
    }

    struct ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
    ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
    struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);
    size_t offset = hn->offset;
    size_t size = ggml_backend_buft_get_alloc_size(buft, node);
    ggml_dyn_tallocr_free_tensor(alloc, offset, size, node);
    hn->allocated = false;
}

static int get_node_buffer_id(const int * node_buffer_ids, int i) {
    return node_buffer_ids ? node_buffer_ids[i] : 0;
}

static void ggml_gallocr_alloc_graph_impl(ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids) {
    // clear hash tables
    memset(galloc->hash_set.keys, 0, galloc->hash_set.size * sizeof(struct ggml_tensor *));
    memset(galloc->hash_values,   0, galloc->hash_set.size * sizeof(struct hash_node));

    // allocate leafs
    // these may be tensors that the application is not using in the graph, but may still want to allocate for other purposes
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        ggml_gallocr_allocate_node(galloc, leaf, get_node_buffer_id(leaf_buffer_ids, i));
    }

    // count number of children and views
    // allocate other graph inputs and leafs first to avoid overwriting them
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];

        // TODO: better way to add external dependencies
        // GGML_OP_NONE does not appear normally in the graph nodes, but is used by ggml-backend to add dependencies to
        // control when some tensors are allocated and freed. in this case, the dependencies are in `src`, but the node
        // itself is never used and should not be considered a dependency
        if (ggml_is_view(node) && node->op != GGML_OP_NONE) {
            struct ggml_tensor * view_src = node->view_src;
            ggml_gallocr_hash_get(galloc, view_src)->n_views += 1;
        }

        if (node->flags & GGML_TENSOR_FLAG_INPUT) {
            ggml_gallocr_allocate_node(galloc, graph->nodes[i], get_node_buffer_id(node_buffer_ids, i));
        }

        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }

            ggml_gallocr_hash_get(galloc, src)->n_children += 1;

            // allocate explicit inputs
            if (src->flags & GGML_TENSOR_FLAG_INPUT) {
                ggml_gallocr_allocate_node(galloc, src, get_node_buffer_id(node_buffer_ids, i));
            }
        }
    }

    // allocate tensors
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int buffer_id = get_node_buffer_id(node_buffer_ids, i);

        // allocate parents (only leafs need to be allocated at this point)
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * parent = node->src[j];
            if (parent == NULL) {
                continue;
            }
            ggml_gallocr_allocate_node(galloc, parent, buffer_id);
        }

        // allocate node
        ggml_gallocr_allocate_node(galloc, node, buffer_id);

        AT_PRINTF("exec: %s (%s) <= ", ggml_op_desc(node), node->name);
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * parent = node->src[j];
            if (parent == NULL) {
                continue;
            }
            AT_PRINTF("%s", parent->name);
            if (j < GGML_MAX_SRC - 1 && node->src[j + 1] != NULL) {
                AT_PRINTF(", ");
            }
        }
        AT_PRINTF("\n");

        // update parents
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * parent = node->src[j];
            if (parent == NULL) {
                continue;
            }
            struct hash_node * p_hn = ggml_gallocr_hash_get(galloc, parent);
            p_hn->n_children -= 1;

            AT_PRINTF("parent %s: %d children, %d views, allocated: %d\n",
                parent->name, p_hn->n_children, p_hn->n_views, p_hn->allocated);

            if (p_hn->n_children == 0 && p_hn->n_views == 0) {
                if (ggml_is_view(parent)) {
                    struct ggml_tensor * view_src = parent->view_src;
                    struct hash_node * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                    view_src_hn->n_views -= 1;
                    AT_PRINTF("view_src %s: %d children, %d views\n",
                        view_src->name, view_src_hn->n_children, view_src_hn->n_views);
                    if (view_src_hn->n_views == 0 && view_src_hn->n_children == 0 && view_src_hn->allocated) {
                        ggml_gallocr_free_node(galloc, view_src, buffer_id);
                    }
                }
                else if (p_hn->allocated) {
                    ggml_gallocr_free_node(galloc, parent, buffer_id);
                }
            }
            AT_PRINTF("\n");
        }
    }
}

bool ggml_gallocr_reserve_n(ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids) {
    size_t hash_size = graph->visited_hash_table.size;

    // initialize hash table
    if (galloc->hash_set.size < hash_size) {
        free(galloc->hash_set.keys);
        free(galloc->hash_values);
        galloc->hash_set.size = hash_size;
        galloc->hash_set.keys = calloc(sizeof(struct ggml_tensor *), hash_size);
        galloc->hash_values   = calloc(sizeof(struct hash_node), hash_size);
        GGML_ASSERT(galloc->hash_set.keys != NULL);
        GGML_ASSERT(galloc->hash_values != NULL);
    } else {
        // reset hash table
        memset(galloc->hash_set.keys, 0, sizeof(struct ggml_tensor *) * galloc->hash_set.size);
        memset(galloc->hash_values,   0, sizeof(struct hash_node) * galloc->hash_set.size);
    }

    // reset allocators
    for (int i = 0; i < galloc->n_buffers; i++) {
        ggml_dyn_tallocr_reset(galloc->buf_tallocs[i]);
    }

    // allocate in hash table
    ggml_gallocr_alloc_graph_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids);

    // set the node_allocs from the hash table
    if (galloc->n_nodes < graph->n_nodes) {
        free(galloc->node_allocs);
        galloc->node_allocs = calloc(sizeof(struct node_alloc), graph->n_nodes);
        GGML_ASSERT(galloc->node_allocs != NULL);
    }
    galloc->n_nodes = graph->n_nodes;
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct node_alloc * node_alloc = &galloc->node_allocs[i];
        node_alloc->buffer_id = get_node_buffer_id(node_buffer_ids, i);
        if (node->view_src || node->data) {
            node_alloc->dst.offset = SIZE_MAX;
            node_alloc->dst.size_max = 0;
        } else {
            struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);
            node_alloc->dst.offset   = hn->offset;
            node_alloc->dst.size_max = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], node);
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (!src || src->view_src || src->data) {
                node_alloc->src[j].offset = SIZE_MAX;
                node_alloc->src[j].size_max = 0;
            } else {
                struct hash_node * hn = ggml_gallocr_hash_get(galloc, src);
                node_alloc->src[j].offset   = hn->offset;
                node_alloc->src[j].size_max = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], src);
            }
        }
    }
    if (galloc->n_leafs < graph->n_leafs) {
        free(galloc->leaf_allocs);
        galloc->leaf_allocs = calloc(sizeof(galloc->leaf_allocs[0]), graph->n_leafs);
        GGML_ASSERT(galloc->leaf_allocs != NULL);
    }
    galloc->n_leafs = graph->n_leafs;
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        struct hash_node * hn = ggml_gallocr_hash_get(galloc, leaf);
        galloc->leaf_allocs[i].buffer_id = hn->buffer_id;
        if (leaf->view_src || leaf->data) {
            galloc->leaf_allocs[i].leaf.offset = SIZE_MAX;
            galloc->leaf_allocs[i].leaf.size_max = 0;
        } else {
            galloc->leaf_allocs[i].leaf.offset = hn->offset;
            galloc->leaf_allocs[i].leaf.size_max = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], leaf);
        }
    }

    // reallocate buffers if needed
    for (int i = 0; i < galloc->n_buffers; i++) {
        size_t cur_size = galloc->buffers[i] ? ggml_backend_buffer_get_size(galloc->buffers[i]) : 0;
        size_t new_size = ggml_dyn_tallocr_max_size(galloc->buf_tallocs[i]);

        // even if there are no tensors allocated in this buffer, we still need to allocate it to initialize views
        if (new_size > cur_size || galloc->buffers[i] == NULL) {
#ifndef NDEBUG
            fprintf(stderr, "%s: reallocating %s buffer from size %.02f MiB to %.02f MiB\n", __func__, ggml_backend_buft_name(galloc->bufts[i]), cur_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            ggml_backend_buffer_free(galloc->buffers[i]);
            galloc->buffers[i] = ggml_backend_buft_alloc_buffer(galloc->bufts[i], new_size);
            if (galloc->buffers[i] == NULL) {
                fprintf(stderr, "%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(galloc->bufts[i]), new_size);
                return false;
            }
        }
    }

    return true;
}

bool ggml_gallocr_reserve(ggml_gallocr_t galloc, struct ggml_cgraph *graph) {
    return ggml_gallocr_reserve_n(galloc, graph, NULL, NULL);
}

static void ggml_gallocr_init_tensor(ggml_gallocr_t galloc, struct ggml_tensor * tensor, int buffer_id, struct tensor_alloc * tensor_alloc) {
    assert(tensor->data || tensor->view_src || ggml_backend_buffer_get_alloc_size(galloc->buffers[buffer_id], tensor) <= tensor_alloc->size_max);

    if (tensor->view_src != NULL) {
        if (tensor->buffer == NULL) {
            assert(tensor_alloc->offset == SIZE_MAX);
            if (tensor->view_src->buffer == NULL) {
                // this tensor was allocated without ggml-backend
                return;
            }
            ggml_backend_view_init(galloc->buffers[buffer_id], tensor);
        }
    } else {
        if (tensor->data == NULL) {
            assert(tensor_alloc->offset != SIZE_MAX);
            assert(ggml_backend_buffer_get_alloc_size(galloc->buffers[buffer_id], tensor) <= tensor_alloc->size_max);
            void * base = ggml_backend_buffer_get_base(galloc->buffers[buffer_id]);
            void * addr = (char *)base + tensor_alloc->offset;
            ggml_backend_tensor_alloc(galloc->buffers[buffer_id], tensor, addr);
        } else {
            if (tensor->buffer == NULL) {
                // this tensor was allocated without ggml-backend
                return;
            }
        }
    }
}

static bool ggml_gallocr_node_needs_realloc(ggml_gallocr_t galloc, struct ggml_tensor * node, struct node_alloc * nalloc, struct tensor_alloc * talloc) {
    ggml_backend_buffer_type_t buft = galloc->bufts[nalloc->buffer_id];
    size_t node_size = (node->data || node->view_src) ? 0 : ggml_backend_buft_get_alloc_size(buft, node);
    return talloc->size_max >= node_size;
}

static bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    if (galloc->n_nodes != graph->n_nodes) {
#ifndef NDEBUG
        fprintf(stderr, "%s: graph has different number of nodes\n", __func__);
#endif
        return true;
    }

    if (galloc->n_leafs != graph->n_leafs) {
#ifndef NDEBUG
        fprintf(stderr, "%s: graph has different number of leafs\n", __func__);
#endif
        return true;
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct node_alloc * node_alloc = &galloc->node_allocs[i];

        if (!ggml_gallocr_node_needs_realloc(galloc, node, node_alloc, &node_alloc->dst)) {
#ifndef NDEBUG
            fprintf(stderr, "%s: node %s is not valid\n", __func__, node->name);
#endif
            return true;
        }

        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            if (!ggml_gallocr_node_needs_realloc(galloc, src, node_alloc, &node_alloc->src[j])) {
#ifndef NDEBUG
                fprintf(stderr, "%s: src %d (%s) of node %s is not valid\n", __func__, j, src->name, node->name);
#endif
                return true;
            }
        }
    }

    return false;
}

bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    if (ggml_gallocr_needs_realloc(galloc, graph)) {
        if (galloc->n_buffers == 1) {
#ifndef NDEBUG
            fprintf(stderr, "%s: reallocating buffers automatically\n", __func__);
#endif
            if (!ggml_gallocr_reserve(galloc, graph)) {
                return false;
            }
        } else {
#ifndef NDEBUG
            fprintf(stderr, "%s: cannot reallocate multi buffer graph automatically, call reserve\n", __func__);
#endif
            return false;
        }
    }

    // reset buffers
    for (int i = 0; i < galloc->n_buffers; i++) {
        if (galloc->buffers[i] != NULL) {
            ggml_backend_buffer_reset(galloc->buffers[i]);
        }
    }

    // allocate the graph tensors from the previous assignments
    // leafs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        struct leaf_alloc * leaf_alloc = &galloc->leaf_allocs[i];
        ggml_gallocr_init_tensor(galloc, leaf, leaf_alloc->buffer_id, &leaf_alloc->leaf);
    }
    // nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct node_alloc * node_alloc = &galloc->node_allocs[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            ggml_gallocr_init_tensor(galloc, src, node_alloc->buffer_id, &node_alloc->src[j]);
        }
        ggml_gallocr_init_tensor(galloc, node, node_alloc->buffer_id, &node_alloc->dst);
    }

    return true;
}

size_t ggml_gallocr_get_buffer_size(ggml_gallocr_t galloc, int buffer_id) {
    GGML_ASSERT(buffer_id >= 0 && buffer_id < galloc->n_buffers);

    if (galloc->buffers[buffer_id] == NULL) {
        return 0;
    }
    return ggml_backend_buffer_get_size(galloc->buffers[buffer_id]);
}

// utils

static bool alloc_tensor_range(struct ggml_context * ctx,
        struct ggml_tensor * first, struct ggml_tensor * last,
        ggml_backend_buffer_type_t buft, size_t size,
        ggml_backend_buffer_t ** buffers, size_t * n_buffers) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, size);
    if (buffer == NULL) {
#ifndef NDEBUG
        fprintf(stderr, "%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), size);
#endif
        for (size_t i = 0; i < *n_buffers; i++) {
            ggml_backend_buffer_free(*buffers[i]);
        }
        free(*buffers);
        return false;
    }

    struct ggml_tallocr tallocr = ggml_tallocr_new(buffer);

    for (struct ggml_tensor * t = first; t != last; t = ggml_get_next_tensor(ctx, t)) {
        if (t->data == NULL) {
            if (t->view_src == NULL) {
                ggml_tallocr_alloc(&tallocr, t);
            } else if (t->buffer == NULL) {
                ggml_backend_view_init(buffer, t);
            }
        } else {
            if (t->view_src != NULL && t->buffer == NULL) {
                // view of a pre-allocated tensor
                ggml_backend_view_init(buffer, t);
            }
        }
    }

    *buffers = realloc(*buffers, sizeof(ggml_backend_buffer_t) * (*n_buffers + 1));
    (*buffers)[(*n_buffers)++] = buffer;

    return true;
}

ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    size_t alignment = ggml_backend_buft_get_alignment(buft);
    size_t max_size = ggml_backend_buft_get_max_size(buft);

    ggml_backend_buffer_t * buffers = NULL;
    size_t n_buffers = 0;

    size_t cur_buf_size = 0;
    struct ggml_tensor * first = ggml_get_first_tensor(ctx);
    for (struct ggml_tensor * t = first; t != NULL; t = ggml_get_next_tensor(ctx, t)) {
        size_t this_size = 0;
        if (t->data == NULL && t->view_src == NULL) {
            this_size = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }

        if (this_size > max_size) {
            fprintf(stderr, "%s: tensor %s is too large to fit in a %s buffer (tensor size: %zu, max buffer size: %zu)\n",
                    __func__, t->name,
                    ggml_backend_buft_name(buft),
                    this_size, max_size);
            for (size_t i = 0; i < n_buffers; i++) {
                ggml_backend_buffer_free(buffers[i]);
            }
            free(buffers);
            return NULL;
        }

        if ((cur_buf_size + this_size) > max_size) {
            // allocate tensors in the current buffer
            if (!alloc_tensor_range(ctx, first, t, buft, cur_buf_size, &buffers, &n_buffers)) {
                return NULL;
            }
            first = t;
            cur_buf_size = this_size;
        } else {
            cur_buf_size += this_size;
        }
    }

    // allocate remaining tensors
    if (cur_buf_size > 0) {
        if (!alloc_tensor_range(ctx, first, NULL, buft, cur_buf_size, &buffers, &n_buffers)) {
            return NULL;
        }
    }

    if (n_buffers == 0) {
#ifndef NDEBUG
        fprintf(stderr, "%s: all tensors in the context are already allocated\n", __func__);
#endif
        return NULL;
    }

    ggml_backend_buffer_t buffer;
    if (n_buffers == 1) {
        buffer = buffers[0];
    } else {
        buffer = ggml_backend_multi_buffer_alloc_buffer(buffers, n_buffers);
    }
    free(buffers);
    return buffer;
}

ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors(struct ggml_context * ctx, ggml_backend_t backend) {
    return ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(backend));
}
