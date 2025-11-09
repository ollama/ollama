# Personal Training Guide: Making Models Smarter & More Like You

## Overview

Your agent system now has **Personal Training Data** that makes models smarter and more personalized. The system automatically learns from your conversations and you can manually add training data to make responses more like you.

## How It Works

### 1. **Automatic Learning** (Already Active)
The system automatically detects:
- **Goals & Projectzs**: When you say "I want to...", "I'm working on...", etc.
- **Values & Beliefs**: When you mention "I believe...", "important to me...", etc.
- **Common Phrases**: Frequently used phrases and expressions
- **Communication Patterns**: Your question style and preferred response length

### 2. **Enhanced Context**
Every agent now receives:
- Your personal facts
- Your values and beliefs
- Your current goals
- Your writing style examples
- Your common phrases
- Your communication patterns

This makes responses more personalized and authentic to your way of thinking.

## Manual Training Data Management

### Using the API

#### Add Training Data
```bash
# Add a writing style example
curl -X POST http://localhost:3000/api/training/add \
  -H "Content-Type: application/json" \
  -d '{"type": "writing_style", "data": "I prefer concise, technical explanations with code examples."}'

# Add a personal fact
curl -X POST http://localhost:3000/api/training/add \
  -H "Content-Type: application/json" \
  -d '{"type": "personal_fact", "data": "I'm a software engineer working on AI systems."}'

# Add a value
curl -X POST http://localhost:3000/api/training/add \
  -H "Content-Type: application/json" \
  -d '{"type": "value", "data": "I value efficiency and clean code over quick hacks."}'

# Add a goal
curl -X POST http://localhost:3000/api/training/add \
  -H "Content-Type: application/json" \
  -d '{"type": "goal", "data": "I want to build a personal AI assistant that understands my coding style."}'

# Add a common phrase
curl -X POST http://localhost:3000/api/training/add \
  -H "Content-Type: application/json" \
  -d '{"type": "common_phrase", "data": "Let me think about this..."}'

# Set expertise level
curl -X POST http://localhost:3000/api/training/add \
  -H "Content-Type: application/json" \
  -d '{"type": "expertise", "data": "{\"domain\": \"JavaScript\", \"level\": \"expert\"}"}'

# Set communication style
curl -X POST http://localhost:3000/api/training/add \
  -H "Content-Type: application/json" \
  -d '{"type": "communication_style", "data": "{\"questionStyle\": \"detailed\", \"responseLength\": \"long\"}"}'
```

#### View Training Data
```bash
curl http://localhost:3000/api/training
```

#### Remove Training Data
```bash
curl -X POST http://localhost:3000/api/training/remove \
  -H "Content-Type: application/json" \
  -d '{"type": "writing_style", "data": "example to remove"}'
```

#### Export Training Data (for Fine-tuning)
```bash
curl http://localhost:3000/api/training/export -o training-data.json
```

## Training Data Types

### 1. **Writing Style Examples**
Examples of how you write. Models will try to match this style.
- **Example**: "I prefer bullet points and code snippets over long paragraphs."
- **Use**: When you want responses to match your writing style

### 2. **Personal Facts**
Key facts about you that agents should know.
- **Example**: "I'm a full-stack developer with 5 years of experience."
- **Use**: For context about your background

### 3. **Values & Beliefs**
What matters to you and your principles.
- **Example**: "I believe in open source and knowledge sharing."
- **Use**: To align responses with your values

### 4. **Goals & Projects**
Current goals and active projects.
- **Example**: "I'm building a multi-agent AI system."
- **Use**: To keep agents aware of your current focus

### 5. **Common Phrases**
Phrases you frequently use.
- **Example**: "Let me check that...", "Actually, I think..."
- **Use**: To make responses feel more natural to you

### 6. **Domain Expertise**
Areas where you're an expert.
- **Example**: `{"domain": "Python", "level": "expert"}`
- **Use**: To adjust technical depth of responses

### 7. **Communication Patterns**
Your preferred communication style.
- **Question Style**: `direct`, `exploratory`, `detailed`
- **Response Length**: `short`, `medium`, `long`

## Fine-Tuning Models (Advanced)

### Step 1: Export Your Training Data
```bash
curl http://localhost:3000/api/training/export -o my-training-data.json
```

### Step 2: Prepare Training Data
The exported JSON contains:
- Your profile and personality
- Conversation history
- Training data
- Summaries

### Step 3: Fine-Tune with Ollama (if supported)
Ollama supports fine-tuning through Modelfiles. You can:
1. Create a Modelfile with your training data
2. Use `ollama create` to create a custom model
3. Set it as the default model for agents

### Step 4: Use Fine-Tuned Model
Update agent model preferences in `server.js` to use your custom model.

## Best Practices

1. **Start with Automatic Learning**: Just chat normally - the system learns automatically
2. **Add Key Facts**: Manually add important facts about yourself
3. **Provide Style Examples**: Add 3-5 examples of your writing style
4. **Update Goals Regularly**: Keep your goals current
5. **Review & Refine**: Check `/api/training` periodically and remove outdated data

## Example: Training Your "Coder" Agent

```bash
# Tell it your coding style
curl -X POST http://localhost:3000/api/training/add \
  -d '{"type": "writing_style", "data": "I write clean, well-commented code with TypeScript."}'

# Set your expertise
curl -X POST http://localhost:3000/api/training/add \
  -d '{"type": "expertise", "data": "{\"domain\": \"TypeScript\", \"level\": \"expert\"}"}'

# Add your current project
curl -X POST http://localhost:3000/api/training/add \
  -d '{"type": "goal", "data": "I'm building a Node.js agent system with Express."}'

# Add your values
curl -X POST http://localhost:3000/api/training/add \
  -d '{"type": "value", "data": "I prioritize maintainability and type safety."}'
```

Now your Coder agent will:
- Write code in your style
- Match your expertise level
- Reference your current project
- Align with your values

## Monitoring Training Data

Check what the system knows about you:
```bash
# View all training data
curl http://localhost:3000/api/training | jq

# View your profile
curl http://localhost:3000/api/memory | jq '.profile.personalTraining'
```

## Next Steps

1. **Chat normally** - The system learns automatically
2. **Add key facts** - Use the API to add important information
3. **Review periodically** - Check what's been learned
4. **Export for fine-tuning** - When ready, export and fine-tune a model

## Tips for Maximum Personalization

- **Be specific**: Instead of "I like code", say "I prefer functional programming with TypeScript"
- **Add context**: Include why something matters to you
- **Update regularly**: Keep goals and projects current
- **Provide examples**: Show, don't just tell - add writing style examples
- **Be consistent**: Use similar language patterns so the system learns your style

---

**Note**: All training data is stored in `data/user-profile.json` and persists across sessions.

