# Advanced Capabilities Guide 🚀

Your agent system now has **powerful advanced capabilities** for code execution, development workflows, and complex task automation!

## 🎯 What's New

### Enhanced Code Execution
- **Multiple Languages**: Python, JavaScript, R, Go, Rust, and more
- **Package Management**: Auto-install dependencies
- **REPL Sessions**: Interactive code execution
- **Advanced Features**: Input/output handling, timeouts, error recovery

### Development Tools
- **Git Operations**: Commit, push, pull, branch management
- **Test Runners**: Jest, pytest, unittest, mocha
- **Code Analysis**: Complexity, quality, security checks
- **Server Management**: Start/stop development servers

### Data & Visualization
- **Database Queries**: SQLite support (extensible)
- **Data Visualization**: Generate charts and graphs
- **File Operations**: Search, replace, batch operations

### Process Management
- **Process Monitoring**: List, kill, monitor system processes
- **Resource Management**: Track resource usage

## 💻 Code Execution Examples

### Basic Code Execution

```javascript
// Python
[TOOL: execute_code {"code": "print('Hello from Python!')", "language": "python"}]

// JavaScript
[TOOL: execute_code {"code": "console.log('Hello from Node!')", "language": "javascript"}]

// Shell
[TOOL: execute_code {"code": "ls -la", "language": "shell"}]
```

### Advanced Code with Packages

```javascript
// Install and use numpy
[TOOL: execute_code_advanced {
  "code": "import numpy as np; arr = np.array([1,2,3]); print(arr.mean())",
  "language": "python",
  "packages": ["numpy"]
}]

// Install and use express
[TOOL: execute_code_advanced {
  "code": "const express = require('express'); console.log('Express loaded');",
  "language": "javascript",
  "packages": ["express"]
}]
```

### Interactive REPL Sessions

```javascript
// Start a REPL session
[TOOL: repl_session {
  "action": "start",
  "language": "python",
  "sessionId": "my-session"
}]

// Execute code in the session
[TOOL: repl_session {
  "action": "execute",
  "sessionId": "my-session",
  "code": "x = 10\ny = 20\nprint(x + y)"
}]

// Stop the session
[TOOL: repl_session {
  "action": "stop",
  "sessionId": "my-session"
}]
```

## 🧪 Testing & Quality

### Run Tests

```javascript
// Jest tests
[TOOL: run_tests {
  "framework": "jest",
  "path": ".",
  "options": ["--coverage"]
}]

// Pytest
[TOOL: run_tests {
  "framework": "pytest",
  "path": "tests/",
  "options": ["-v", "--cov"]
}]

// Mocha
[TOOL: run_tests {
  "framework": "mocha",
  "path": "test/",
  "options": ["--reporter", "spec"]
}]
```

### Code Analysis

```javascript
// Analyze code quality
[TOOL: analyze_code {
  "filepath": "src/main.js",
  "language": "javascript",
  "analysis": "all"
}]

// Returns:
// - Lines of code
// - Cyclomatic complexity
// - Security issues
// - Code quality metrics
```

## 📦 Package Management

```javascript
// Install npm packages
[TOOL: package_manager {
  "manager": "npm",
  "action": "install",
  "packages": ["express", "axios", "lodash"]
}]

// Install Python packages
[TOOL: package_manager {
  "manager": "pip3",
  "action": "install",
  "packages": ["numpy", "pandas", "matplotlib"]
}]

// List installed packages
[TOOL: package_manager {
  "manager": "npm",
  "action": "list"
}]
```

## 🗄️ Database Operations

```javascript
// Query SQLite database
[TOOL: database_query {
  "database": "data/users.db",
  "query": "SELECT * FROM users WHERE age > 18",
  "type": "sqlite"
}]

// Create table
[TOOL: database_query {
  "database": "data/app.db",
  "query": "CREATE TABLE IF NOT EXISTS posts (id INTEGER PRIMARY KEY, title TEXT)",
  "type": "sqlite"
}]
```

## 🔧 Git Operations

```javascript
// Check status
[TOOL: git_operation {
  "operation": "status",
  "args": []
}]

// Commit changes
[TOOL: git_operation {
  "operation": "commit",
  "args": ["-a", "-m", "Add new feature"]
}]

// View log
[TOOL: git_operation {
  "operation": "log",
  "args": ["--oneline", "-10"]
}]

// Create branch
[TOOL: git_operation {
  "operation": "checkout",
  "args": ["-b", "feature/new-feature"]
}]
```

## 🖥️ Server Management

```javascript
// Start Node.js server
[TOOL: start_server {
  "type": "node",
  "port": 3000,
  "script": "server.js",
  "options": []
}]

// Start Python HTTP server
[TOOL: start_server {
  "type": "python",
  "port": 8000,
  "options": []
}]

// Start Go server
[TOOL: start_server {
  "type": "go",
  "script": "main.go",
  "options": []
}]
```

## 📊 Data Visualization

```javascript
// Create line chart
[TOOL: create_visualization {
  "data": "{\"labels\": [\"Jan\", \"Feb\", \"Mar\"], \"datasets\": [{\"data\": [10, 20, 30]}]}",
  "type": "line",
  "options": {"title": "Sales Data"}
}]

// Create bar chart from CSV
[TOOL: create_visualization {
  "data": "name,value\nA,10\nB,20\nC,30",
  "type": "bar",
  "options": {}
}]
```

## 📁 Advanced File Operations

```javascript
// Search for text in files
[TOOL: file_operations {
  "operation": "search",
  "pattern": "TODO",
  "directory": "src/"
}]

// Replace text in files
[TOOL: file_operations {
  "operation": "replace",
  "pattern": "old-text",
  "replacement": "new-text",
  "directory": "src/"
}]

// Find files by pattern
[TOOL: file_operations {
  "operation": "find",
  "pattern": "*.test.js",
  "directory": "."
}]
```

## 🔍 Process Management

```javascript
// List all processes
[TOOL: process_manager {
  "action": "list"
}]

// Find processes by name
[TOOL: process_manager {
  "action": "list",
  "name": "node"
}]

// Kill a process
[TOOL: process_manager {
  "action": "kill",
  "pid": 1234
}]
```

## 🎨 Real-World Examples

### Example 1: Data Analysis Pipeline

```javascript
// 1. Install packages
[TOOL: package_manager {
  "manager": "pip3",
  "action": "install",
  "packages": ["pandas", "matplotlib"]
}]

// 2. Analyze data
[TOOL: execute_code_advanced {
  "code": "import pandas as pd; df = pd.read_csv('data.csv'); print(df.describe())",
  "language": "python",
  "packages": ["pandas"]
}]

// 3. Create visualization
[TOOL: create_visualization {
  "data": "...",
  "type": "line"
}]
```

### Example 2: Full-Stack Development

```javascript
// 1. Install dependencies
[TOOL: package_manager {
  "manager": "npm",
  "action": "install",
  "packages": ["express", "cors", "dotenv"]
}]

// 2. Write server code
[TOOL: write_file {
  "filepath": "server.js",
  "content": "const express = require('express'); const app = express(); app.listen(3000);"
}]

// 3. Run tests
[TOOL: run_tests {
  "framework": "jest",
  "path": "."
}]

// 4. Start server
[TOOL: start_server {
  "type": "node",
  "port": 3000,
  "script": "server.js"
}]
```

### Example 3: Machine Learning Workflow

```javascript
// 1. Install ML packages
[TOOL: package_manager {
  "manager": "pip3",
  "action": "install",
  "packages": ["scikit-learn", "numpy", "pandas"]
}]

// 2. Train model
[TOOL: execute_code_advanced {
  "code": `
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test)}")
  `,
  "language": "python",
  "packages": ["scikit-learn", "numpy", "pandas"]
}]
```

## 🚀 Creative Use Cases

### 1. Automated Code Review
- Analyze code quality
- Run tests
- Check for security issues
- Generate report

### 2. Data Science Workflow
- Load and clean data
- Run analysis
- Generate visualizations
- Export results

### 3. DevOps Automation
- Run tests
- Build applications
- Deploy to servers
- Monitor processes

### 4. Learning & Experimentation
- Try new libraries
- Test code snippets
- Explore APIs
- Build prototypes

### 5. Code Generation
- Generate boilerplate
- Create test files
- Set up project structure
- Initialize frameworks

## ⚠️ Security & Best Practices

### Security Features
- ✅ Path sanitization (prevents directory traversal)
- ✅ Command filtering (blocks dangerous commands)
- ✅ Timeout limits (prevents hanging)
- ✅ Isolated execution (temporary files)
- ✅ Resource limits (memory, CPU)

### Best Practices
1. **Always validate inputs** before using tools
2. **Use timeouts** for long-running operations
3. **Clean up** temporary files and sessions
4. **Monitor** resource usage
5. **Test** code in isolated environments

## 🎓 Learning Resources

### Try These Challenges:

1. **Build a Data Pipeline**
   - Read CSV files
   - Process data
   - Generate visualizations
   - Export results

2. **Create a REST API**
   - Set up Express
   - Create endpoints
   - Write tests
   - Start server

3. **Machine Learning Project**
   - Load dataset
   - Train model
   - Evaluate performance
   - Visualize results

4. **Full-Stack App**
   - Frontend (HTML/CSS/JS)
   - Backend (Node.js)
   - Database (SQLite)
   - Tests

## 🔮 Future Enhancements

Potential additions:
- Docker container execution
- Cloud deployment tools
- Advanced debugging
- Performance profiling
- Code generation from descriptions
- AI model training workflows
- Real-time collaboration
- Version control integration

## 📚 API Reference

All tools are available via:
- **Direct tool calls** in agent responses: `[TOOL: toolname {params}]`
- **API endpoint**: `POST /api/tools/execute`
- **Workflow integration**: Use in workflow steps

See `COMPLEX_TASKS_GUIDE.md` for workflow examples.

---

**Ready to build amazing things?** Start by asking an agent to execute some code or run a test! 🎉

