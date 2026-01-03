// query.go provides flexible query building with adjustable parameters.
package sqlite

import (
	"database/sql"
	"fmt"
	"strings"
)

// QueryBuilder constructs SQL queries with adjustable parameters.
type QueryBuilder struct {
	db     *sql.DB
	tables []tableRef
	cols   []string
	joins  []joinClause
	wheres []whereClause
	orders []string
	limit  int
	offset int
	params []interface{}
}

type tableRef struct {
	name  string
	alias string
}

type joinClause struct {
	joinType string // JOIN, LEFT JOIN, etc.
	table    string
	alias    string
	on       string
}

type whereClause struct {
	condition string
	args      []interface{}
}

// Query creates a new query builder on the given database.
func Query(db *sql.DB) *QueryBuilder {
	return &QueryBuilder{
		db:     db,
		params: make([]interface{}, 0),
	}
}

// From sets the primary table(s) for the query.
func (q *QueryBuilder) From(table string, alias ...string) *QueryBuilder {
	a := ""
	if len(alias) > 0 {
		a = alias[0]
	}
	q.tables = append(q.tables, tableRef{name: table, alias: a})
	return q
}

// Select specifies columns to select.
func (q *QueryBuilder) Select(cols ...string) *QueryBuilder {
	q.cols = append(q.cols, cols...)
	return q
}

// Join adds a JOIN clause.
func (q *QueryBuilder) Join(table, alias, on string) *QueryBuilder {
	q.joins = append(q.joins, joinClause{
		joinType: "JOIN",
		table:    table,
		alias:    alias,
		on:       on,
	})
	return q
}

// LeftJoin adds a LEFT JOIN clause.
func (q *QueryBuilder) LeftJoin(table, alias, on string) *QueryBuilder {
	q.joins = append(q.joins, joinClause{
		joinType: "LEFT JOIN",
		table:    table,
		alias:    alias,
		on:       on,
	})
	return q
}

// Where adds a WHERE condition with parameters.
func (q *QueryBuilder) Where(condition string, args ...interface{}) *QueryBuilder {
	q.wheres = append(q.wheres, whereClause{condition: condition, args: args})
	return q
}

// OrderBy adds ORDER BY clauses.
func (q *QueryBuilder) OrderBy(orders ...string) *QueryBuilder {
	q.orders = append(q.orders, orders...)
	return q
}

// Limit sets the LIMIT clause.
func (q *QueryBuilder) Limit(n int) *QueryBuilder {
	q.limit = n
	return q
}

// Offset sets the OFFSET clause.
func (q *QueryBuilder) Offset(n int) *QueryBuilder {
	q.offset = n
	return q
}

// Build constructs the SQL query string and parameters.
func (q *QueryBuilder) Build() (string, []interface{}) {
	var sb strings.Builder
	var params []interface{}

	// SELECT
	sb.WriteString("SELECT ")
	if len(q.cols) == 0 {
		sb.WriteString("*")
	} else {
		sb.WriteString(strings.Join(q.cols, ", "))
	}

	// FROM
	sb.WriteString(" FROM ")
	for i, t := range q.tables {
		if i > 0 {
			sb.WriteString(", ")
		}
		sb.WriteString(t.name)
		if t.alias != "" {
			sb.WriteString(" AS ")
			sb.WriteString(t.alias)
		}
	}

	// JOINs
	for _, j := range q.joins {
		sb.WriteString(" ")
		sb.WriteString(j.joinType)
		sb.WriteString(" ")
		sb.WriteString(j.table)
		if j.alias != "" {
			sb.WriteString(" AS ")
			sb.WriteString(j.alias)
		}
		sb.WriteString(" ON ")
		sb.WriteString(j.on)
	}

	// WHERE
	if len(q.wheres) > 0 {
		sb.WriteString(" WHERE ")
		for i, w := range q.wheres {
			if i > 0 {
				sb.WriteString(" AND ")
			}
			sb.WriteString("(")
			sb.WriteString(w.condition)
			sb.WriteString(")")
			params = append(params, w.args...)
		}
	}

	// ORDER BY
	if len(q.orders) > 0 {
		sb.WriteString(" ORDER BY ")
		sb.WriteString(strings.Join(q.orders, ", "))
	}

	// LIMIT
	if q.limit > 0 {
		sb.WriteString(fmt.Sprintf(" LIMIT %d", q.limit))
	}

	// OFFSET
	if q.offset > 0 {
		sb.WriteString(fmt.Sprintf(" OFFSET %d", q.offset))
	}

	return sb.String(), params
}

// Execute runs the query and returns rows.
func (q *QueryBuilder) Execute() (*sql.Rows, error) {
	query, params := q.Build()
	return q.db.Query(query, params...)
}

// ExecuteOne runs the query expecting a single row.
func (q *QueryBuilder) ExecuteOne() *sql.Row {
	query, params := q.Build()
	return q.db.QueryRow(query, params...)
}

// --- Predefined query templates ---

// QueryTemplate is a parameterized query that can be executed with different values.
type QueryTemplate struct {
	db       *sql.DB
	template string
	defaults map[string]interface{}
}

// NewTemplate creates a query template with named parameters.
// Use :name syntax for named parameters in the template.
// Example: "SELECT * FROM tensors WHERE layer = :layer AND component = :component"
func NewTemplate(db *sql.DB, template string) *QueryTemplate {
	return &QueryTemplate{
		db:       db,
		template: template,
		defaults: make(map[string]interface{}),
	}
}

// Default sets a default value for a named parameter.
func (t *QueryTemplate) Default(name string, value interface{}) *QueryTemplate {
	t.defaults[name] = value
	return t
}

// Execute runs the template with the given parameters.
// Parameters override defaults.
func (t *QueryTemplate) Execute(params map[string]interface{}) (*sql.Rows, error) {
	query, args := t.expand(params)
	return t.db.Query(query, args...)
}

// ExecuteOne runs the template expecting a single row.
func (t *QueryTemplate) ExecuteOne(params map[string]interface{}) *sql.Row {
	query, args := t.expand(params)
	return t.db.QueryRow(query, args...)
}

func (t *QueryTemplate) expand(params map[string]interface{}) (string, []interface{}) {
	// Merge defaults with provided params
	merged := make(map[string]interface{})
	for k, v := range t.defaults {
		merged[k] = v
	}
	for k, v := range params {
		merged[k] = v
	}

	// Replace :name with ? and collect args in order
	query := t.template
	var args []interface{}

	for name, value := range merged {
		placeholder := ":" + name
		if strings.Contains(query, placeholder) {
			query = strings.Replace(query, placeholder, "?", 1)
			args = append(args, value)
		}
	}

	return query, args
}

// --- Common query patterns for model data ---

// TensorQuery provides common tensor lookup patterns.
type TensorQuery struct {
	db *sql.DB
}

// NewTensorQuery creates a tensor query helper.
func NewTensorQuery(db *sql.DB) *TensorQuery {
	return &TensorQuery{db: db}
}

// ByLayer returns tensors for a specific layer.
func (tq *TensorQuery) ByLayer(layer int) (*sql.Rows, error) {
	return tq.db.Query(
		"SELECT id, name, dims, dtype, n_elements, byte_size FROM tensors WHERE layer = ?",
		layer,
	)
}

// ByComponent returns tensors matching a component pattern.
func (tq *TensorQuery) ByComponent(component string) (*sql.Rows, error) {
	return tq.db.Query(
		"SELECT id, name, dims, dtype, n_elements, byte_size FROM tensors WHERE component = ?",
		component,
	)
}

// ByName returns a tensor by exact name.
func (tq *TensorQuery) ByName(name string) *sql.Row {
	return tq.db.QueryRow(
		"SELECT id, name, dims, dtype, n_elements, byte_size FROM tensors WHERE name = ?",
		name,
	)
}

// InLayers returns tensors in a range of layers.
func (tq *TensorQuery) InLayers(start, end int) (*sql.Rows, error) {
	return tq.db.Query(
		"SELECT id, name, layer, dims, dtype, n_elements, byte_size FROM tensors WHERE layer >= ? AND layer <= ? ORDER BY layer",
		start, end,
	)
}

// VocabQuery provides common vocabulary lookup patterns.
type VocabQuery struct {
	db *sql.DB
}

// NewVocabQuery creates a vocabulary query helper.
func NewVocabQuery(db *sql.DB) *VocabQuery {
	return &VocabQuery{db: db}
}

// ByTokenID returns the token string for an ID.
func (vq *VocabQuery) ByTokenID(id int) (string, error) {
	var s string
	err := vq.db.QueryRow("SELECT token_string FROM vocab WHERE token_id = ?", id).Scan(&s)
	return s, err
}

// ByTokenString returns the token ID for a string.
func (vq *VocabQuery) ByTokenString(s string) (int, error) {
	var id int
	err := vq.db.QueryRow("SELECT token_id FROM vocab WHERE token_string = ?", s).Scan(&id)
	if err == sql.ErrNoRows {
		return -1, nil
	}
	return id, err
}

// InRange returns tokens in an ID range.
func (vq *VocabQuery) InRange(start, end int) (*sql.Rows, error) {
	return vq.db.Query(
		"SELECT token_id, token_string FROM vocab WHERE token_id >= ? AND token_id <= ? ORDER BY token_id",
		start, end,
	)
}

// MatchPrefix returns tokens starting with a prefix.
func (vq *VocabQuery) MatchPrefix(prefix string) (*sql.Rows, error) {
	return vq.db.Query(
		"SELECT token_id, token_string FROM vocab WHERE token_string LIKE ? ORDER BY token_id",
		prefix+"%",
	)
}
