package llama

type node[T any] struct {
	t    T
	next *node[T]
	prev *node[T]
}

type deque[T any] struct {
	head     *node[T]
	tail     *node[T]
	size     int
	capacity int
}

func (d *deque[T]) Empty() bool {
	return d.size == 0
}

func (d *deque[T]) Len() int {
	return d.size
}

func (d *deque[T]) Cap() int {
	return d.capacity
}

func (d *deque[T]) Push(t T) {
	if d.capacity > 0 && d.size >= d.capacity {
		d.PopLeft()
	}

	n := node[T]{t: t}
	if d.head != nil {
		n.next = d.head
		d.head.prev = &n
		d.head = &n
	} else {
		d.head = &n
		d.tail = &n
	}

	d.size++
}

func (d *deque[T]) PushLeft(t T) {
	if d.capacity > 0 && d.size >= d.capacity {
		d.Pop()
	}

	n := node[T]{t: t}
	if d.tail != nil {
		n.prev = d.tail
		d.tail.next = &n
		d.tail = &n
	} else {
		d.head = &n
		d.tail = &n
	}

	d.size++
}

func (d *deque[T]) Pop() *T {
	if d.Empty() {
		return nil
	}

	head := d.head
	d.head = head.next
	if d.head != nil {
		d.head.prev = nil
	} else {
		d.tail = nil
	}

	d.size--
	return &head.t
}

func (d *deque[T]) PopLeft() *T {
	if d.Empty() {
		return nil
	}

	tail := d.tail
	d.tail = tail.prev
	if d.tail != nil {
		d.tail.next = nil
	} else {
		d.head = nil
	}

	d.size--
	return &tail.t
}

func (d *deque[T]) Data() (data []T) {
	for n := d.head; n != nil; n = n.next {
		data = append(data, n.t)
	}

	return data
}
