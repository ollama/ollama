;; Simple SMT verification for GPU selection
(set-logic QF_LIA)

;; GPU scoring function
(declare-fun gpu_score (Int Int Bool) Int)

;; Score = memory_gb * 10 + tflops + (tensor_cores ? 50 : 0)
(assert (forall ((mem Int) (tflops Int) (tensor Bool))
  (= (gpu_score mem tflops tensor)
     (+ (* mem 10) tflops (ite tensor 50 0)))))

;; Test cases from our implementation
(assert (= (gpu_score 8 20 true) 150))   ;; RTX 3070
(assert (= (gpu_score 12 17 true) 187))  ;; Arc B580
(assert (= (gpu_score 12 25 false) 145)) ;; RX 6700 XT

;; Arc B580 should have highest score
(assert (> (gpu_score 12 17 true) (gpu_score 8 20 true)))
(assert (> (gpu_score 12 17 true) (gpu_score 12 25 false)))

(check-sat)
(get-model)