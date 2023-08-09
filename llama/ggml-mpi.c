//go:build mpi

/**
 * llama.cpp - git 8183159cf3def112f6d1fe94815fce70e1bffa12
 *
 * MIT License
 *
 * Copyright (c) 2023 Georgi Gerganov
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "ggml-mpi.h"

#include "ggml.h"

#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define UNUSED GGML_UNUSED

struct ggml_mpi_context {
    int rank;
    int size;
};

void ggml_mpi_backend_init(void) {
    MPI_Init(NULL, NULL);
}

void ggml_mpi_backend_free(void) {
    MPI_Finalize();
}

struct ggml_mpi_context * ggml_mpi_init(void) {
    struct ggml_mpi_context * ctx = calloc(1, sizeof(struct ggml_mpi_context));

    MPI_Comm_rank(MPI_COMM_WORLD, &ctx->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ctx->size);

    return ctx;
}

void ggml_mpi_free(struct ggml_mpi_context * ctx) {
    free(ctx);
}

int ggml_mpi_rank(struct ggml_mpi_context * ctx) {
    return ctx->rank;
}

void ggml_mpi_eval_init(
        struct ggml_mpi_context * ctx_mpi,
                            int * n_tokens,
                            int * n_past,
                            int * n_threads) {
    UNUSED(ctx_mpi);

    // synchronize the worker node parameters with the root node
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(n_tokens,  1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(n_past,    1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(n_threads, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

static int ggml_graph_get_node_idx(struct ggml_cgraph * gf, const char * name) {
    struct ggml_tensor * t = ggml_graph_get_tensor(gf, name);
    if (t == NULL) {
        fprintf(stderr, "%s: tensor %s not found\n", __func__, name);
        return -1;
    }

    for (int i = 0; i < gf->n_nodes; i++) {
        if (gf->nodes[i] == t) {
            return i;
        }
    }

    fprintf(stderr, "%s: tensor %s not found in graph (should not happen)\n", __func__, name);
    return -1;
}

static void ggml_mpi_tensor_send(struct ggml_tensor * t, int mpi_rank_dst) {
    MPI_Datatype mpi_type;

    switch (t->type) {
        case GGML_TYPE_I32: mpi_type = MPI_INT32_T; break;
        case GGML_TYPE_F32: mpi_type = MPI_FLOAT;   break;
        default: GGML_ASSERT(false && "not implemented");
    }

    const int retval = MPI_Send(t->data, ggml_nelements(t), mpi_type, mpi_rank_dst, 0, MPI_COMM_WORLD);
    GGML_ASSERT(retval == MPI_SUCCESS);
}

static void ggml_mpi_tensor_recv(struct ggml_tensor * t, int mpi_rank_src) {
    MPI_Datatype mpi_type;

    switch (t->type) {
        case GGML_TYPE_I32: mpi_type = MPI_INT32_T; break;
        case GGML_TYPE_F32: mpi_type = MPI_FLOAT;   break;
        default: GGML_ASSERT(false && "not implemented");
    }

    MPI_Status status; UNUSED(status);

    const int retval = MPI_Recv(t->data, ggml_nelements(t), mpi_type, mpi_rank_src, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    GGML_ASSERT(retval == MPI_SUCCESS);
}

// TODO: there are many improvements that can be done to this implementation
void ggml_mpi_graph_compute_pre(
        struct ggml_mpi_context * ctx_mpi,
             struct ggml_cgraph * gf,
                            int   n_layers) {
    const int mpi_rank = ctx_mpi->rank;
    const int mpi_size = ctx_mpi->size;

    struct ggml_tensor * inp_tokens = ggml_graph_get_tensor(gf, "inp_tokens");
    if (inp_tokens == NULL) {
        fprintf(stderr, "%s: tensor 'inp_tokens' not found\n", __func__);
        return;
    }

    struct ggml_tensor * inp0 = ggml_graph_get_tensor(gf, "layer_inp_0");
    if (inp0 == NULL) {
        fprintf(stderr, "%s: tensor 'inp0' not found\n", __func__);
        return;
    }

    GGML_ASSERT(inp0 == gf->nodes[0]);

    // distribute the compute graph into slices across the MPI nodes
    //
    // the main node (0) processes the last layers + the remainder of the compute graph
    // and is responsible to pass the input tokens to the first node (1)
    //
    // node 1:   [(  0) * n_per_node, (  1) * n_per_node)
    // node 2:   [(  1) * n_per_node, (  2) * n_per_node)
    // ...
    // node n-1: [(n-2) * n_per_node, (n-1) * n_per_node)
    // node 0:   [(n-1) * n_per_node,            n_nodes)
    //
    if (mpi_rank > 0) {
        if (mpi_rank == 1) {
            // the first node (1) receives the input tokens from the main node (0)
            ggml_mpi_tensor_recv(inp_tokens, 0);
        } else {
            // recv input data for each node into the "inp0" tensor (i.e. the first node in the compute graph)
            ggml_mpi_tensor_recv(inp0, mpi_rank - 1);
        }
    } else if (mpi_size > 1) {
        // node 0 sends the input tokens to node 1
        ggml_mpi_tensor_send(inp_tokens, 1);

        // recv the output data from the last node
        ggml_mpi_tensor_recv(inp0, mpi_size - 1);
    }

    {
        const int n_per_node = (n_layers + (mpi_size - 1)) / mpi_size;

        const int mpi_idx = mpi_rank > 0 ? mpi_rank - 1 : mpi_size - 1;

        const int il0 =               (mpi_idx + 0) * n_per_node;
        const int il1 = MIN(n_layers, (mpi_idx + 1) * n_per_node);

        char name_l0[GGML_MAX_NAME];
        char name_l1[GGML_MAX_NAME];

        snprintf(name_l0, sizeof(name_l0), "layer_inp_%d", il0);
        snprintf(name_l1, sizeof(name_l1), "layer_inp_%d", il1);

        const int idx_l0 =                ggml_graph_get_node_idx(gf, name_l0);
        const int idx_l1 = mpi_rank > 0 ? ggml_graph_get_node_idx(gf, name_l1) + 1 : gf->n_nodes;

        if (idx_l0 < 0 || idx_l1 < 0) {
            fprintf(stderr, "%s: layer input nodes not found\n", __func__);
            return;
        }

        // attach the input data to all nodes that need it
        // TODO: not great - should be able to do this without modifying the compute graph (see next TODO below)
        for (int i = idx_l0; i < idx_l1; i++) {
            if (gf->nodes[i]->src[0] == gf->nodes[idx_l0]) {
                gf->nodes[i]->src[0] =  inp0;
            }
            if (gf->nodes[i]->src[1] == gf->nodes[idx_l0]) {
                gf->nodes[i]->src[1] =  inp0;
            }
        }

        // TODO: instead of rearranging the nodes, we should be able to execute a subset of the compute graph
        for (int i = 1; i < idx_l1 - idx_l0; i++) {
            gf->nodes[i] = gf->nodes[idx_l0 + i];
            gf->grads[i] = gf->grads[idx_l0 + i];
        }

        // the first node performs the "get_rows" operation, the rest of the nodes get the data from the previous node
        if (mpi_idx != 0) {
            gf->nodes[0]->op = GGML_OP_NONE;
        }

        gf->n_nodes = idx_l1 - idx_l0;

        //fprintf(stderr, "%s: node %d: processing %d nodes [%d, %d)\n", __func__, mpi_rank, gf->n_nodes, il0, il1);
    }
}

void ggml_mpi_graph_compute_post(
        struct ggml_mpi_context * ctx_mpi,
             struct ggml_cgraph * gf,
                            int   n_layers) {
    UNUSED(n_layers);

    const int mpi_rank = ctx_mpi->rank;
    const int mpi_size = ctx_mpi->size;

    // send the output data to the next node
    if (mpi_rank > 0) {
        ggml_mpi_tensor_send(gf->nodes[gf->n_nodes - 1], (mpi_rank + 1) % mpi_size);
    }
}
