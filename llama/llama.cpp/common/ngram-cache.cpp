#include "ngram-cache.h"
#include "common.h"
#include "log.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <thread>
#include <algorithm>

void common_ngram_cache_update(common_ngram_cache & ngram_cache, int ngram_min, int ngram_max,
                              std::vector<llama_token> & inp, int nnew, bool print_progress) {
    const int64_t t_start_ms = ggml_time_ms();
    const int64_t inp_size = inp.size();

    const int64_t n_todo = inp_size * (ngram_max - ngram_min + 1);
    int64_t n_done = 0;

    for (int64_t ngram_size = ngram_min; ngram_size <= ngram_max; ++ngram_size) {
        const int64_t i_start = std::max(inp_size - nnew, ngram_size);
        for (int64_t i = i_start; i < inp_size; ++i) {
            const int64_t ngram_start = i - ngram_size;
            common_ngram ngram(&inp[ngram_start], ngram_size);
            const llama_token token = inp[i];

            common_ngram_cache::iterator part_it = ngram_cache.find(ngram);
            if (part_it == ngram_cache.end()) {
                common_ngram_cache_part part;
                part.emplace(token, 1);
                ngram_cache.emplace(ngram, part);
            } else {
                common_ngram_cache_part::iterator token_count_it = part_it->second.find(token);
                if (token_count_it == part_it->second.end()) {
                    part_it->second.emplace(token, 1);
                } else {
                    token_count_it->second++;
                }
            }
            ++n_done;

            if (print_progress && n_done % 10000000 == 0) {
                const int64_t t_now_ms = ggml_time_ms();
                const int64_t eta_ms   = (inp_size*(ngram_max-ngram_min+1) - n_done) * (t_now_ms - t_start_ms) / n_done;
                const int64_t eta_min  = eta_ms / (60*1000);
                const int64_t eta_s    = (eta_ms - 60*1000*eta_min) / 1000;

                fprintf(stderr, "%s: %" PRId64 "/%" PRId64 " done, ETA: %02" PRId64 ":%02" PRId64 "\n", __func__, n_done, n_todo, eta_min, eta_s);
            }
        }
    }
}

// Helper function to get a token from the combined, speculative sequence of inp and draft.
static llama_token get_token(const std::vector<llama_token> & inp, const std::vector<llama_token> & draft, const size_t i) {
    return i < inp.size() ? inp[i] : draft[1 + i - inp.size()];
}

// If sample size or percentage are below these thresholds the draft is aborted early:
constexpr int    draft_min_sample_size_lax[LLAMA_NGRAM_MAX] = { 2,  2,  1,  1};
constexpr int        draft_min_percent_lax[LLAMA_NGRAM_MAX] = {66, 50, 50, 50};
constexpr int draft_min_sample_size_strict[LLAMA_NGRAM_MAX] = { 4,  3,  2,  2};
constexpr int     draft_min_percent_strict[LLAMA_NGRAM_MAX] = {75, 66, 66, 66};

// Helper function that tries to draft a token from only the static ngram cache:
static llama_token try_draft(common_ngram_cache & nc_static, const common_ngram ngram_static) {
    common_ngram_cache::iterator part_static_it = nc_static.find(ngram_static);
    if (part_static_it == nc_static.end()) {
        return LLAMA_TOKEN_NULL;
    }
    const common_ngram_cache_part part_static = part_static_it->second;

    int max_count_static  = 0;
    int sum_count_static  = 0;
    llama_token max_token = LLAMA_TOKEN_NULL;

    for (std::pair<llama_token, int> token_count_static : part_static) {
        const llama_token token = token_count_static.first;
        const int32_t count_static  = token_count_static.second;

        if (count_static > max_count_static) {
            max_token        = token;
            max_count_static = count_static;
        }
        sum_count_static += count_static;
    }

    if (sum_count_static < draft_min_sample_size_lax[LLAMA_NGRAM_STATIC-1]) {
        return LLAMA_TOKEN_NULL;
    }
    if (100*max_count_static < draft_min_percent_lax[LLAMA_NGRAM_STATIC-1]*sum_count_static) {
        return LLAMA_TOKEN_NULL;
    }
    return max_token;
}

// Try to draft a token from primary cache (context/dynamic), validate with static cache:
static llama_token try_draft(
    common_ngram_cache & nc_primary, const std::vector<common_ngram> & ngrams_primary, common_ngram_cache_part & part_static,
    const int * min_sample_size, const int * min_percent) {

    llama_token drafted_token = LLAMA_TOKEN_NULL;

    for (int i = ngrams_primary.size()-1; i >= 0 && drafted_token == LLAMA_TOKEN_NULL; --i) {
        const common_ngram ngram_primary = ngrams_primary[i];

        common_ngram_cache::iterator part_primary_it = nc_primary.find(ngram_primary);
        if (part_primary_it == nc_primary.end()) {
            continue;
        }
        const common_ngram_cache_part part_primary = part_primary_it->second;

        int max_count_primary = 0;
        int max_count_static  = 0;
        int sum_count_primary = 0;
        llama_token max_token = LLAMA_TOKEN_NULL;

        for (std::pair<llama_token, int> token_count_primary : part_primary) {
            const llama_token token = token_count_primary.first;

            common_ngram_cache_part::iterator token_count_static_it = part_static.find(token);

            const int32_t count_primary = token_count_primary.second;
            const int32_t count_static  = token_count_static_it != part_static.end() ? 100*token_count_static_it->second : 1;

            if (count_primary*count_static > max_count_primary*max_count_static) {
                max_token         = token;
                max_count_primary = count_primary;
                max_count_static  = count_static;
            }
            sum_count_primary += count_primary;
        }

        if (sum_count_primary < min_sample_size[i]) {
            continue;
        }
        if (100*max_count_primary < min_percent[i]*sum_count_primary) {
            continue;;
        }
        drafted_token = max_token;
    }

    return drafted_token;
}

void common_ngram_cache_draft(
    std::vector<llama_token> & inp, std::vector<llama_token> & draft, int n_draft, int ngram_min, int ngram_max,
    common_ngram_cache & nc_context, common_ngram_cache & nc_dynamic, common_ngram_cache & nc_static
) {
    GGML_ASSERT(draft.size() == 1);
    const int inp_size = inp.size();

    if (inp_size < LLAMA_NGRAM_STATIC) {
        return;
    }

    while ((int) draft.size()-1 < n_draft) {
        llama_token drafted_token = LLAMA_TOKEN_NULL;

        const int ngram_start_static = inp_size-LLAMA_NGRAM_STATIC + draft.size()-1;
        common_ngram ngram_static;
        for (int j = ngram_start_static; j < ngram_start_static + LLAMA_NGRAM_STATIC; ++j) {
            ngram_static.tokens[j-ngram_start_static] = get_token(inp, draft, j);
        }
        common_ngram_cache::iterator part_static_it = nc_static.find(ngram_static);
        common_ngram_cache_part part_static;
        if (part_static_it != nc_static.end()) {
            part_static = part_static_it->second;
        }

        // cd = context + dynamic
        std::vector<common_ngram> ngrams_cd;
        for (int ngram_size_cd = ngram_min; ngram_size_cd <= ngram_max; ++ngram_size_cd) {
            const int ngram_start_cd = inp_size-ngram_size_cd + draft.size()-1;
            common_ngram ngram_cd;
            for (int j = ngram_start_cd; j < ngram_start_cd + ngram_size_cd; ++j) {
                ngram_cd.tokens[j-ngram_start_cd] = get_token(inp, draft, j);
            }
            ngrams_cd.push_back(ngram_cd);
        }
        if (drafted_token == LLAMA_TOKEN_NULL) {
            drafted_token = try_draft(nc_context, ngrams_cd, part_static, draft_min_sample_size_lax, draft_min_percent_lax);
        }
        if (drafted_token == LLAMA_TOKEN_NULL) {
            drafted_token = try_draft(nc_dynamic, ngrams_cd, part_static, draft_min_sample_size_strict, draft_min_percent_strict);
        }
        if (drafted_token == LLAMA_TOKEN_NULL) {
            drafted_token = try_draft(nc_static, ngram_static);
        }

        if (drafted_token == LLAMA_TOKEN_NULL) {
            break;
        }

        LOG(" - draft candidate: token=%d\n", drafted_token);
        draft.push_back(drafted_token);
    }
}

void common_ngram_cache_save(common_ngram_cache & ngram_cache, std::string & filename) {
    std::ofstream file_out(filename, std::ios::binary);
    for (std::pair<common_ngram, common_ngram_cache_part> item : ngram_cache) {
        const common_ngram      ngram        = item.first;
        common_ngram_cache_part token_counts = item.second;
        GGML_ASSERT(!token_counts.empty());
        const int32_t ntokens = token_counts.size();
        GGML_ASSERT(ntokens > 0);

        file_out.write(reinterpret_cast<const char *>(&ngram),   sizeof(common_ngram));
        file_out.write(reinterpret_cast<const char *>(&ntokens), sizeof(int32_t));
        for (std::pair<llama_token, int32_t> item2 : token_counts) {
            const llama_token token = item2.first;
            const int32_t     count = item2.second;
            GGML_ASSERT(count > 0);

            file_out.write(reinterpret_cast<const char *>(&token), sizeof(llama_token));
            file_out.write(reinterpret_cast<const char *>(&count), sizeof(int32_t));
        }
    }

}

common_ngram_cache common_ngram_cache_load(std::string & filename) {
    std::ifstream hashmap_file(filename, std::ios::binary);
    if (!hashmap_file) {
        throw std::ifstream::failure("Unable to open file " + filename);
    }
    common_ngram_cache ngram_cache;

    common_ngram ngram;
    int32_t     ntokens;
    llama_token token;
    int32_t     count;

    char * ngramc   = reinterpret_cast<char*>(&ngram);
    char * ntokensc = reinterpret_cast<char*>(&ntokens);
    char * tokenc   = reinterpret_cast<char*>(&token);
    char * countc   = reinterpret_cast<char*>(&count);
    while(hashmap_file.read(ngramc, sizeof(common_ngram))) {
        GGML_ASSERT(!hashmap_file.eof());
        GGML_ASSERT(hashmap_file.read(ntokensc, sizeof(int32_t)));
        GGML_ASSERT(ntokens > 0);
        common_ngram_cache_part token_counts;

        for (int i = 0; i < ntokens; ++i) {
            GGML_ASSERT(!hashmap_file.eof());
            GGML_ASSERT(hashmap_file.read(tokenc, sizeof(llama_token)));
            GGML_ASSERT(!hashmap_file.eof());
            GGML_ASSERT(hashmap_file.read(countc, sizeof(int32_t)));
            GGML_ASSERT(count > 0);
            token_counts.emplace(token, count);
        }

        ngram_cache.emplace(ngram, token_counts);
    }
    GGML_ASSERT(hashmap_file.eof());

    return ngram_cache;
}

void common_ngram_cache_merge(common_ngram_cache & ngram_cache_target, common_ngram_cache & ngram_cache_add) {
    for (std::pair<common_ngram, common_ngram_cache_part> ngram_part : ngram_cache_add) {
        const common_ngram      ngram = ngram_part.first;
        common_ngram_cache_part  part = ngram_part.second;

        common_ngram_cache::iterator part_merged_it = ngram_cache_target.find(ngram);
        if (part_merged_it == ngram_cache_target.end()) {
            ngram_cache_target.emplace(ngram, part);
            continue;
        }

        for (std::pair<llama_token, int32_t> token_count : part) {
            const llama_token token = token_count.first;
            const int32_t     count = token_count.second;
            GGML_ASSERT(count > 0);

            common_ngram_cache_part::iterator token_count_merged_it = part_merged_it->second.find(token);
            if (token_count_merged_it == part_merged_it->second.end()) {
                part_merged_it->second.emplace(token, count);
                continue;
            }

            token_count_merged_it->second += count;
        }
    }
}
