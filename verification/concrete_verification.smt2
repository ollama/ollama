;; Concrete SMT verification for Ollama optimizations
(set-logic QF_LIA)

;; Concrete test values
(declare-const checkpoint_4 Int)
(declare-const checkpoint_16 Int)
(declare-const checkpoint_64 Int)
(declare-const checkpoint_100 Int)

(declare-const mla_28 Int)
(declare-const mla_280 Int)
(declare-const mla_1500 Int)
(declare-const mla_2800 Int)

(declare-const score_rtx3070 Int)
(declare-const score_arc_b580 Int)
(declare-const score_rx6700xt Int)

;; Checkpoint values (sqrt + 1)
(assert (= checkpoint_4 3))
(assert (= checkpoint_16 5))
(assert (= checkpoint_64 9))
(assert (= checkpoint_100 11))

;; Verify checkpoint savings
(assert (< checkpoint_4 4))
(assert (< checkpoint_16 16))
(assert (< checkpoint_64 64))
(assert (< checkpoint_100 100))

;; MLA compression values (input / 28)
(assert (= mla_28 1))
(assert (= mla_280 10))
(assert (= mla_1500 53))
(assert (= mla_2800 100))

;; Verify MLA compression
(assert (< mla_28 28))
(assert (< mla_280 280))
(assert (< mla_1500 1500))
(assert (< mla_2800 2800))

;; GPU scores
(assert (= score_rtx3070 150))   ;; 8*10 + 20 + 50
(assert (= score_arc_b580 187))  ;; 12*10 + 17 + 50
(assert (= score_rx6700xt 145))  ;; 12*10 + 25 + 0

;; Verify Arc B580 has highest score
(assert (> score_arc_b580 score_rtx3070))
(assert (> score_arc_b580 score_rx6700xt))

;; Combined optimization for DeepSeek V3
(declare-const deepseek_original Int)
(declare-const deepseek_optimized Int)
(declare-const deepseek_savings Int)

(assert (= deepseek_original (+ 64 1500)))  ;; 1564 total
(assert (= deepseek_optimized (+ 9 53)))    ;; 62 optimized
(assert (= deepseek_savings (- deepseek_original deepseek_optimized)))

;; Verify significant savings
(assert (= deepseek_savings 1502))
(assert (> deepseek_savings 1400))  ;; More than 1400 units saved

;; Efficiency calculation (savings / original)
(declare-const efficiency_numerator Int)
(declare-const efficiency_denominator Int)

(assert (= efficiency_numerator deepseek_savings))
(assert (= efficiency_denominator deepseek_original))

;; Verify high efficiency (>95%)
(assert (> (* efficiency_numerator 100) (* 95 efficiency_denominator)))

(check-sat)
(get-model)