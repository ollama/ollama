package ordered

type Map[K comparable, V any] struct {
	s []K
	m map[K]V
}

func NewMap[K comparable, V any]() *Map[K, V] {
	return &Map[K, V]{
		s: make([]K, 0),
		m: make(map[K]V),
	}
}

type iter_Seq2[K, V any] func(func(K, V) bool)

func (m *Map[K, V]) Items() iter_Seq2[K, V] {
	return func(yield func(K, V) bool) {
		for _, k := range m.s {
			if !yield(k, m.m[k]) {
				return
			}
		}
	}
}

func (m *Map[K, V]) Add(k K, v V) {
	if _, ok := m.m[k]; !ok {
		m.s = append(m.s, k)
		m.m[k] = v
	}
}
