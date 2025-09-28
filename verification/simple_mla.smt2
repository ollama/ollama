;; Simple SMT verification for MLA compression
(set-logic QF_LIA)

;; MLA compression function: kv / 28
(declare-fun mla_compress (Int) Int)

;; Define compression
(assert (forall ((kv Int))
  (=> (>= kv 0)
      (= (mla_compress kv) (div kv 28)))))

;; Test cases
(assert (= (mla_compress 28) 1))
(assert (= (mla_compress 280) 10))
(assert (= (mla_compress 1500) 53))
(assert (= (mla_compress 2800) 100))

;; Compression property
(assert (< (mla_compress 1500) 1500))
(assert (< (mla_compress 2800) 2800))

(check-sat)
(get-model)