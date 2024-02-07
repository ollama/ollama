#include "ext_server.h"

// Necessary evil since the server types are not defined in a header
#include "server.cpp"

// Low level API access to verify GPU access
#if defined(GGML_USE_CUBLAS)
#if defined(GGML_USE_HIPBLAS)
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hip/hip_fp16.h>
#ifdef __HIP_PLATFORM_AMD__
// for rocblas_initialize()
#include "rocblas/rocblas.h"
#endif // __HIP_PLATFORM_AMD__
#define cudaGetDevice hipGetDevice
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#else
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#endif // defined(GGML_USE_HIPBLAS)
#endif // GGML_USE_CUBLAS

// Expose the llama server as a callable extern "C" API
llama_server_context *llama = NULL;
std::thread ext_server_thread;

void llama_server_init(ext_server_params *sparams, ext_server_resp_t *err) {
  assert(err != NULL && sparams != NULL);
  log_set_target(stderr);
  if (!sparams->verbose_logging) {
    server_verbose = true;
    log_disable();
  }

  LOG_TEE("system info: %s\n", llama_print_system_info());
  err->id = 0;
  err->msg[0] = '\0';
  try {
    llama = new llama_server_context;
    gpt_params params;
    params.n_ctx = sparams->n_ctx;
    params.n_batch = sparams->n_batch;
    if (sparams->n_threads > 0) {
      params.n_threads = sparams->n_threads;
    }
    params.n_parallel = sparams->n_parallel;
    params.rope_freq_base = sparams->rope_freq_base;
    params.rope_freq_scale = sparams->rope_freq_scale;

    if (sparams->memory_f16) {
      params.cache_type_k = "f16";
      params.cache_type_v = "f16";
    } else {
      params.cache_type_k = "f32";
      params.cache_type_v = "f32";
    }

    params.n_gpu_layers = sparams->n_gpu_layers;
    params.main_gpu = sparams->main_gpu;
    params.use_mlock = sparams->use_mlock;
    params.use_mmap = sparams->use_mmap;
    params.numa = sparams->numa;
    params.embedding = sparams->embedding;
    if (sparams->model != NULL) {
      params.model = sparams->model;
    }

    if (sparams->lora_adapters != NULL) {
      for (ext_server_lora_adapter *la = sparams->lora_adapters; la != NULL;
          la = la->next) {
        params.lora_adapter.push_back(std::make_tuple(la->adapter, la->scale));
      }

      params.use_mmap = false;
    }

    if (sparams->mmproj != NULL) {
      params.mmproj = std::string(sparams->mmproj);
    }

#if defined(GGML_USE_CUBLAS)
    // Before attempting to init the backend which will assert on error, verify the CUDA/ROCM GPU is accessible
    LOG_TEE("Performing pre-initialization of GPU\n");
    int id;
    cudaError_t cudaErr = cudaGetDevice(&id);
    if (cudaErr != cudaSuccess) {
      err->id = -1;
      snprintf(err->msg, err->msg_len, "Unable to init GPU: %s", cudaGetErrorString(cudaErr));
      return;
    }
#endif

    llama_backend_init(params.numa);

    // load the model
    if (!llama->load_model(params)) {
      // TODO - consider modifying the logging logic or patching load_model so
      // we can capture more detailed error messages and pass them back to the
      // caller for better UX
      err->id = -1;
      snprintf(err->msg, err->msg_len, "error loading model %s",
               params.model.c_str());
      return;
    }

    llama->initialize();
  } catch (std::exception &e) {
    err->id = -1;
    snprintf(err->msg, err->msg_len, "exception %s", e.what());
  } catch (...) {
    err->id = -1;
    snprintf(err->msg, err->msg_len,
             "Unknown exception initializing llama server");
  }
}

void llama_server_start() {
  assert(llama != NULL);
  // TODO mutex to protect thread creation
  ext_server_thread = std::thread([&]() {
    try {
      LOG_TEE("llama server main loop starting\n");
      ggml_time_init();
      llama->queue_tasks.on_new_task(std::bind(
        &llama_server_context::process_single_task, llama, std::placeholders::_1));
      llama->queue_tasks.on_finish_multitask(std::bind(
          &llama_server_context::on_finish_multitask, llama, std::placeholders::_1));
      llama->queue_tasks.on_all_tasks_finished(std::bind(
          &llama_server_context::run_on_all_tasks_finished, llama));
      llama->queue_results.on_multitask_update(std::bind(
          &llama_server_queue::update_multitask,
          &llama->queue_tasks,
          std::placeholders::_1,
          std::placeholders::_2,
          std::placeholders::_3
        ));
      llama->queue_tasks.start_loop();
    } catch (std::exception &e) {
      LOG_TEE("caught exception in llama server main loop: %s\n", e.what());
    } catch (...) {
      LOG_TEE("caught unknown exception in llama server main loop\n");
    }
    LOG_TEE("\nllama server shutting down\n");
    llama_backend_free();
  });
}

void llama_server_stop() {
  assert(llama != NULL);
  LOG_TEE("\ninitiating shutdown - draining remaining tasks...\n");
  // This may take a while for any pending tasks to drain
  // TODO - consider a timeout to cancel tasks if it's taking too long
  llama->queue_tasks.terminate();
  ext_server_thread.join();
  delete llama;
  llama = NULL;
  LOG_TEE("llama server shutdown complete\n");
}

void llama_server_completion(const char *json_req, ext_server_resp_t *resp) {
  assert(llama != NULL && json_req != NULL && resp != NULL);
  resp->id = -1;
  resp->msg[0] = '\0';
  try {
    json data = json::parse(json_req);
    resp->id = llama->queue_tasks.get_new_id();
    llama->queue_results.add_waiting_task_id(resp->id);
    llama->request_completion(resp->id, data, false, false, -1);
  } catch (std::exception &e) {
    snprintf(resp->msg, resp->msg_len, "exception %s", e.what());
  } catch (...) {
    snprintf(resp->msg, resp->msg_len, "Unknown exception during completion");
  }
}

void llama_server_completion_next_result(const int task_id,
                                         ext_server_task_result_t *resp) {
  assert(llama != NULL && resp != NULL);
  std::string msg;
  resp->id = -1;
  resp->stop = false;
  resp->error = false;
  resp->json_resp = NULL;
  std::string result_json;
  try {
    task_result result = llama->queue_results.recv(task_id);
    result_json =
        result.result_json.dump(-1, ' ', false, json::error_handler_t::replace);
    resp->id = result.id;
    resp->stop = result.stop;
    resp->error = result.error;
    if (result.error) {
      LOG_TEE("next result cancel on error\n");
      llama->request_cancel(task_id);
      LOG_TEE("next result removing waiting tak ID: %d\n", task_id);
      llama->queue_results.remove_waiting_task_id(task_id);
    } else if (result.stop) {
      LOG_TEE("next result cancel on stop\n");
      llama->request_cancel(task_id);
      LOG_TEE("next result removing waiting task ID: %d\n", task_id);
      llama->queue_results.remove_waiting_task_id(task_id);
    }
  } catch (std::exception &e) {
    resp->error = true;
    resp->id = -1;
    result_json = "{\"error\":\"exception " + std::string(e.what()) + "\"}";
    LOG_TEE("llama server completion exception %s\n", e.what());
  } catch (...) {
    resp->error = true;
    resp->id = -1;
    result_json = "{\"error\":\"Unknown exception during completion\"}";
    LOG_TEE("llama server completion unknown exception\n");
  }
  const std::string::size_type size = result_json.size() + 1;
  resp->json_resp = new char[size];
  snprintf(resp->json_resp, size, "%s", result_json.c_str());
}

void llama_server_release_task_result(ext_server_task_result_t *result) {
  if (result == NULL || result->json_resp == NULL) {
    return;
  }
  delete[] result->json_resp;
}

void llama_server_completion_cancel(const int task_id, ext_server_resp_t *err) {
  assert(llama != NULL && err != NULL);
  err->id = 0;
  err->msg[0] = '\0';
  try {
    llama->request_cancel(task_id);
    llama->queue_results.remove_waiting_task_id(task_id);
  } catch (std::exception &e) {
    err->id = -1;
    snprintf(err->msg, err->msg_len, "exception %s", e.what());
  } catch (...) {
    err->id = -1;
    snprintf(err->msg, err->msg_len,
             "Unknown exception completion cancel in llama server");
  }
}

void llama_server_tokenize(const char *json_req, char **json_resp,
                           ext_server_resp_t *err) {
  assert(llama != NULL && json_req != NULL && json_resp != NULL && err != NULL);
  *json_resp = NULL;
  err->id = 0;
  err->msg[0] = '\0';
  try {
    const json body = json::parse(json_req);
    std::vector<llama_token> tokens;
    if (body.count("content") != 0) {
      tokens = llama->tokenize(body["content"], false);
    }
    const json data = format_tokenizer_response(tokens);
    std::string result_json = data.dump();
    const std::string::size_type size = result_json.size() + 1;
    *json_resp = new char[size];
    snprintf(*json_resp, size, "%s", result_json.c_str());
  } catch (std::exception &e) {
    err->id = -1;
    snprintf(err->msg, err->msg_len, "exception %s", e.what());
  } catch (...) {
    err->id = -1;
    snprintf(err->msg, err->msg_len, "Unknown exception during tokenize");
  }
}

void llama_server_release_json_resp(char **json_resp) {
  if (json_resp == NULL || *json_resp == NULL) {
    return;
  }
  delete[] *json_resp;
}

void llama_server_detokenize(const char *json_req, char **json_resp,
                             ext_server_resp_t *err) {
  assert(llama != NULL && json_req != NULL && json_resp != NULL && err != NULL);
  *json_resp = NULL;
  err->id = 0;
  err->msg[0] = '\0';
  try {
    const json body = json::parse(json_req);
    std::string content;
    if (body.count("tokens") != 0) {
      const std::vector<llama_token> tokens = body["tokens"];
      content = tokens_to_str(llama->ctx, tokens.cbegin(), tokens.cend());
    }
    const json data = format_detokenized_response(content);
    std::string result_json = data.dump();
    const std::string::size_type size = result_json.size() + 1;
    *json_resp = new char[size];
    snprintf(*json_resp, size, "%s", result_json.c_str());
  } catch (std::exception &e) {
    err->id = -1;
    snprintf(err->msg, err->msg_len, "exception %s", e.what());
  } catch (...) {
    err->id = -1;
    snprintf(err->msg, err->msg_len, "Unknown exception during detokenize");
  }
}

void llama_server_embedding(const char *json_req, char **json_resp,
                            ext_server_resp_t *err) {
  assert(llama != NULL && json_req != NULL && json_resp != NULL && err != NULL);
  *json_resp = NULL;
  err->id = 0;
  err->msg[0] = '\0';
  try {
    const json body = json::parse(json_req);
    json prompt;
    if (body.count("content") != 0) {
      prompt = body["content"];
    } else {
      prompt = "";
    }
    const int task_id = llama->queue_tasks.get_new_id();
    llama->queue_results.add_waiting_task_id(task_id);
    llama->request_completion(task_id, {{"prompt", prompt}, {"n_predict", 0}}, false, true, -1);
    task_result result = llama->queue_results.recv(task_id);
    std::string result_json = result.result_json.dump();
    const std::string::size_type size = result_json.size() + 1;
    *json_resp = new char[size];
    snprintf(*json_resp, size, "%s", result_json.c_str());
    llama->queue_results.remove_waiting_task_id(task_id);
  } catch (std::exception &e) {
    err->id = -1;
    snprintf(err->msg, err->msg_len, "exception %s", e.what());
  } catch (...) {
    err->id = -1;
    snprintf(err->msg, err->msg_len, "Unknown exception during embedding");
  }
}