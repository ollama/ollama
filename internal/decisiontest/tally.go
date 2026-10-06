package decisiontest

import "fmt"

// Tally keeps failed requests in the denominator, including malformed responses.
type Tally struct {
	Requests         int `json:"attempted_requests"`
	Errors           int `json:"errors"`
	CorrectCases     int `json:"correct_cases"`
	Questions        int `json:"questions"`
	CorrectQuestions int `json:"correct_questions"`
}

func (t *Tally) Add(c Case, wrong []string, err error) {
	t.Requests++
	t.Questions += len(c.Expected)
	if err != nil {
		t.Errors++
		return
	}
	t.CorrectQuestions += len(c.Expected) - len(wrong)
	if len(wrong) == 0 {
		t.CorrectCases++
	}
}

func (t Tally) CaseAccuracy() float64 {
	return 100 * float64(t.CorrectCases) / float64(max(t.Requests, 1))
}

func (t Tally) QuestionAccuracy() float64 {
	return 100 * float64(t.CorrectQuestions) / float64(max(t.Questions, 1))
}

func (t Tally) String() string {
	return fmt.Sprintf("cases=%d/%d (%.2f%%), questions=%d/%d (%.2f%%), errors=%d", t.CorrectCases, t.Requests, t.CaseAccuracy(), t.CorrectQuestions, t.Questions, t.QuestionAccuracy(), t.Errors)
}
