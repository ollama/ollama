#include "ext_server.h"
#include <atomic>

#include "server.cpp"

// Expose the llama server as a callable extern "C" API
llama_server_context *llama = NULL;
std::thread ext_server_thread;
bool shutting_down = false;
std::atomic_int recv_counter;

// RAII wrapper for tracking in-flight recv calls
class atomicRecv {
  public:
    atomicRecv(std::atomic<int> &atomic) : atomic(atomic) {
      ++this->atomic;
    }
    ~atomicRecv() {
      --this->atomic;
    }
  private:
    std::atomic<int> &atomic;
};
 
void llama_server_init(ext_server_params *sparams, ext_server_resp_t *err) {
  recv_counter = 0;
  assert(err != NULL && sparams != NULL);

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

    params.n_gpu_layers = sparams->n_gpu_layers;
    params.use_mlock = sparams->use_mlock;
    params.use_mmap = sparams->use_mmap;
    params.embedding = sparams->embedding;
    if (sparams->model != NULL) {
      params.model = sparams->model;
    }

    model_init_backend();

  if (!llama->load_model(params)) { 
    // an error occurred that was not thrown
    err->id = -1;
    snprintf(err->msg, err->msg_len, "error loading model %s", params.model.c_str());
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
      LOG_INFO("llama server main loop starting\n", {});
      ne_time_init();
      llama->queue_tasks.on_new_task(std::bind(
        &llama_server_context::process_single_task, llama, std::placeholders::_1));
      llama->queue_tasks.on_finish_multitask(std::bind(
        &llama_server_context::on_finish_multitask, llama, std::placeholders::_1));
      llama->queue_tasks.on_run_slots(std::bind(
        &llama_server_context::update_slots, llama));
      llama->queue_results.on_multitask_update(std::bind(
          &llama_server_queue::update_multitask,
          &llama->queue_tasks,
          std::placeholders::_1,
          std::placeholders::_2,
          std::placeholders::_3
        ));
      llama->queue_tasks.start_loop();
    } catch (std::exception &e) {
      //LOG_INFO("caught exception in llama server main loop: %s\n", e.what());
    } catch (...) {
      LOG_INFO("caught unknown exception in llama server main loop\n", {});
    }
    LOG_INFO("\nllama server shutting down\n", {});
    if (llama->ctx) model_free(llama->ctx);
  });
}

void llama_server_stop() {
  assert(llama != NULL);
  // Shutdown any in-flight requests and block incoming requests.
  LOG_INFO("\ninitiating shutdown - draining remaining tasks...\n", {});
  shutting_down = true;

  while (recv_counter.load() > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  // This may take a while for any pending tasks to drain
  // TODO - consider a timeout to cancel tasks if it's taking too long
  llama->queue_tasks.terminate();
  ext_server_thread.join();
  delete llama;
  llama = NULL;
  LOG_INFO("llama server shutdown complete\n", {});
  shutting_down = false;
}

void llama_server_completion(const char *json_req, ext_server_resp_t *resp) {
  assert(llama != NULL && json_req != NULL && resp != NULL);
  resp->id = -1;
  resp->msg[0] = '\0';
  try {
    if (shutting_down) {
      throw std::runtime_error("server shutting down");
    }
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
  resp->id = -1;
  resp->stop = false;
  resp->error = false;
  resp->json_resp = NULL;
  std::string result_json;
  try {
    atomicRecv ar(recv_counter);
    task_result result = llama->queue_results.recv(task_id);
    result_json =
        result.result_json.dump(-1, ' ', false, json::error_handler_t::replace);
    resp->id = result.id;
    resp->stop = result.stop;
    resp->error = result.error;
    if (result.error) {
      LOG_INFO("next result cancel on error\n", {});
      llama->request_cancel(task_id);
      //LOG_INFO("next result removing waiting tak ID: %d\n", task_id);
      llama->queue_results.remove_waiting_task_id(task_id);
    } else if (result.stop) {
      LOG_INFO("next result cancel on stop\n", {});
      llama->request_cancel(task_id);
      //LOG_INFO("next result removing waiting task ID: %d\n", task_id);
      llama->queue_results.remove_waiting_task_id(task_id);
    } else if (shutting_down) {
      //LOG_INFO("aborting completion due to shutdown %d\n", task_id);
      llama->request_cancel(task_id);
      llama->queue_results.remove_waiting_task_id(task_id);
      resp->stop = true;
    }
  } catch (std::exception &e) {
    resp->error = true;
    resp->id = -1;
    result_json = "{\"error\":\"exception " + std::string(e.what()) + "\"}";
    //LOG_INFO("llama server completion exception %s\n", e.what());
  } catch (...) {
    resp->error = true;
    resp->id = -1;
    result_json = "{\"error\":\"Unknown exception during completion\"}";
    LOG_INFO("llama server completion unknown exception\n", {});
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
    if (shutting_down) {
      throw std::runtime_error("server shutting down");
    }
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
    if (shutting_down) {
      throw std::runtime_error("server shutting down");
    }
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
    if (shutting_down) {
      throw std::runtime_error("server shutting down");
    }
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
    atomicRecv ar(recv_counter);
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

