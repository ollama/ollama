;; SMT/Z3 Verification for GPU Device Selection
;; Based on proof/verified_gpu_backend.v: device_selection_sound theorem

(set-logic QF_NIA)
(set-info :source |Ollama GPU Device Selection Verification|)

;; GPU device properties
(declare-sort GPU)
(declare-fun memory_size_gb (GPU) Int)
(declare-fun peak_tflops_fp32 (GPU) Int)
(declare-fun supports_tensor_cores (GPU) Bool)
(declare-fun device_score (GPU) Int)

;; Device scoring function: memory_gb * 10 + tflops + tensor_bonus
(assert (forall ((gpu GPU))
  (= (device_score gpu)
     (+ (* (memory_size_gb gpu) 10)
        (peak_tflops_fp32 gpu)
        (ite (supports_tensor_cores gpu) 50 0)))))

;; Memory constraint satisfaction
(declare-fun satisfies_memory_constraint (GPU Int) Bool)
(assert (forall ((gpu GPU) (required_memory Int))
  (= (satisfies_memory_constraint gpu required_memory)
     (>= (memory_size_gb gpu) required_memory))))

;; Device selection function properties
(declare-fun is_best_device (GPU Int) Bool)
(assert (forall ((selected_gpu GPU) (required_memory Int))
  (=> (is_best_device selected_gpu required_memory)
      (satisfies_memory_constraint selected_gpu required_memory))))

;; Optimality property: if a device is selected, no other valid device has higher score
(assert (forall ((selected_gpu GPU) (other_gpu GPU) (required_memory Int))
  (=> (and (is_best_device selected_gpu required_memory)
           (satisfies_memory_constraint other_gpu required_memory))
      (>= (device_score selected_gpu) (device_score other_gpu)))))

;; Define specific GPU devices from our tests
(declare-const rtx_3070 GPU)
(declare-const arc_b580 GPU)
(declare-const rx_6700xt GPU)

;; RTX 3070 specifications
(assert (= (memory_size_gb rtx_3070) 8))
(assert (= (peak_tflops_fp32 rtx_3070) 20))
(assert (= (supports_tensor_cores rtx_3070) true))

;; Intel Arc B580 specifications
(assert (= (memory_size_gb arc_b580) 12))
(assert (= (peak_tflops_fp32 arc_b580) 17))
(assert (= (supports_tensor_cores arc_b580) true))

;; AMD RX 6700 XT specifications
(assert (= (memory_size_gb rx_6700xt) 12))
(assert (= (peak_tflops_fp32 rx_6700xt) 25))
(assert (= (supports_tensor_cores rx_6700xt) false))

;; Calculated scores
(assert (= (device_score rtx_3070) 150))   ;; 8*10 + 20 + 50 = 150
(assert (= (device_score arc_b580) 187))   ;; 12*10 + 17 + 50 = 187
(assert (= (device_score rx_6700xt) 145))  ;; 12*10 + 25 + 0 = 145

;; Test cases for device selection
(declare-const memory_req_4gb Int)
(declare-const memory_req_8gb Int)
(declare-const memory_req_10gb Int)
(declare-const memory_req_15gb Int)

(assert (= memory_req_4gb 4))
(assert (= memory_req_8gb 8))
(assert (= memory_req_10gb 10))
(assert (= memory_req_15gb 15))

;; For 4GB requirement: Arc B580 should be best (highest score among all valid)
(assert (satisfies_memory_constraint rtx_3070 memory_req_4gb))
(assert (satisfies_memory_constraint arc_b580 memory_req_4gb))
(assert (satisfies_memory_constraint rx_6700xt memory_req_4gb))
(assert (is_best_device arc_b580 memory_req_4gb))

;; For 8GB requirement: Arc B580 should still be best
(assert (satisfies_memory_constraint rtx_3070 memory_req_8gb))
(assert (satisfies_memory_constraint arc_b580 memory_req_8gb))
(assert (satisfies_memory_constraint rx_6700xt memory_req_8gb))
(assert (is_best_device arc_b580 memory_req_8gb))

;; For 10GB requirement: Only Arc B580 and RX 6700 XT qualify, Arc B580 wins
(assert (not (satisfies_memory_constraint rtx_3070 memory_req_10gb)))
(assert (satisfies_memory_constraint arc_b580 memory_req_10gb))
(assert (satisfies_memory_constraint rx_6700xt memory_req_10gb))
(assert (is_best_device arc_b580 memory_req_10gb))

;; For 15GB requirement: No device qualifies
(assert (not (satisfies_memory_constraint rtx_3070 memory_req_15gb)))
(assert (not (satisfies_memory_constraint arc_b580 memory_req_15gb)))
(assert (not (satisfies_memory_constraint rx_6700xt memory_req_15gb)))

;; Score ordering properties
(assert (> (device_score arc_b580) (device_score rtx_3070)))
(assert (> (device_score arc_b580) (device_score rx_6700xt)))

;; Tensor core bonus verification
(declare-fun tensor_core_bonus (GPU) Int)
(assert (forall ((gpu GPU))
  (= (tensor_core_bonus gpu)
     (ite (supports_tensor_cores gpu) 50 0))))

(assert (= (tensor_core_bonus rtx_3070) 50))
(assert (= (tensor_core_bonus arc_b580) 50))
(assert (= (tensor_core_bonus rx_6700xt) 0))

;; Memory efficiency considerations
(declare-fun memory_to_compute_ratio (GPU) Real)
(assert (forall ((gpu GPU))
  (=> (> (peak_tflops_fp32 gpu) 0)
      (= (memory_to_compute_ratio gpu)
         (/ (to_real (memory_size_gb gpu)) (to_real (peak_tflops_fp32 gpu)))))))

;; Arc B580 should have good memory-to-compute ratio
(assert (>= (memory_to_compute_ratio arc_b580) 0.5))

;; Check satisfiability
(check-sat)
(get-model)

;; Verification outputs
(echo "RTX 3070 score:")
(eval (device_score rtx_3070))
(echo "Arc B580 score:")
(eval (device_score arc_b580))
(echo "RX 6700 XT score:")
(eval (device_score rx_6700xt))
(echo "Arc B580 satisfies 10GB requirement:")
(eval (satisfies_memory_constraint arc_b580 memory_req_10gb))
(echo "RTX 3070 satisfies 10GB requirement:")
(eval (satisfies_memory_constraint rtx_3070 memory_req_10gb))