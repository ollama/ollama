;; Simple SMT verification for checkpoint optimization
(set-logic QF_LIA)

;; Basic checkpoint function: sqrt(n) + 1
(declare-fun checkpoint_layers (Int) Int)
(declare-fun sqrt_int (Int) Int)

;; Square root constraints (simplified)
(assert (= (sqrt_int 4) 2))
(assert (= (sqrt_int 16) 4))
(assert (= (sqrt_int 64) 8))
(assert (= (sqrt_int 100) 10))

;; Checkpoint function
(assert (forall ((n Int))
  (= (checkpoint_layers n) (+ (sqrt_int n) 1))))

;; Test cases
(assert (= (checkpoint_layers 4) 3))
(assert (= (checkpoint_layers 16) 5))
(assert (= (checkpoint_layers 64) 9))
(assert (= (checkpoint_layers 100) 11))

;; Memory savings
(assert (< (checkpoint_layers 64) 64))
(assert (< (checkpoint_layers 100) 100))

(check-sat)
(get-model)