#!/usr/bin/env node

/**
 * Personal Training Data Manager
 * 
 * Simple CLI tool to manage your personal training data
 * 
 * Usage:
 *   node train-me.js add writing_style "I prefer concise explanations"
 *   node train-me.js add personal_fact "I'm a software engineer"
 *   node train-me.js list
 *   node train-me.js remove writing_style "example"
 *   node train-me.js export
 */

const axios = require('axios');
const fs = require('fs');
const path = require('path');

const API_URL = process.env.AGENT_API_URL || 'http://localhost:3000';

const TYPES = {
    'writing_style': 'Writing Style Example',
    'personal_fact': 'Personal Fact',
    'value': 'Value or Belief',
    'goal': 'Goal or Project',
    'common_phrase': 'Common Phrase',
    'expertise': 'Domain Expertise (JSON: {"domain": "Python", "level": "expert"})',
    'communication_style': 'Communication Style (JSON: {"questionStyle": "direct", "responseLength": "medium"})'
};

async function addTrainingData(type, data) {
    try {
        const response = await axios.post(`${API_URL}/api/training/add`, {
            type,
            data
        });
        console.log(`✅ Added ${TYPES[type] || type}: ${data}`);
        return response.data;
    } catch (error) {
        console.error(`❌ Error: ${error.response?.data?.error || error.message}`);
        process.exit(1);
    }
}

async function removeTrainingData(type, data) {
    try {
        const response = await axios.post(`${API_URL}/api/training/remove`, {
            type,
            data
        });
        console.log(`✅ Removed ${TYPES[type] || type}: ${data}`);
        return response.data;
    } catch (error) {
        console.error(`❌ Error: ${error.response?.data?.error || error.message}`);
        process.exit(1);
    }
}

async function listTrainingData() {
    try {
        const response = await axios.get(`${API_URL}/api/training`);
        const data = response.data;
        
        console.log('\n📚 YOUR PERSONAL TRAINING DATA\n');
        console.log('═'.repeat(60));
        
        if (data.writingStyleExamples?.length > 0) {
            console.log('\n✍️  Writing Style Examples:');
            data.writingStyleExamples.forEach((ex, i) => {
                console.log(`   ${i + 1}. "${ex}"`);
            });
        }
        
        if (data.personalFacts?.length > 0) {
            console.log('\n👤 Personal Facts:');
            data.personalFacts.forEach((fact, i) => {
                console.log(`   ${i + 1}. ${fact}`);
            });
        }
        
        if (data.valuesAndBeliefs?.length > 0) {
            console.log('\n💎 Values & Beliefs:');
            data.valuesAndBeliefs.forEach((value, i) => {
                console.log(`   ${i + 1}. ${value}`);
            });
        }
        
        if (data.goalsAndProjects?.length > 0) {
            console.log('\n🎯 Goals & Projects:');
            data.goalsAndProjects.forEach((goal, i) => {
                console.log(`   ${i + 1}. ${goal}`);
            });
        }
        
        if (data.communicationPatterns?.commonPhrases?.length > 0) {
            console.log('\n💬 Common Phrases:');
            data.communicationPatterns.commonPhrases.forEach((phrase, i) => {
                console.log(`   ${i + 1}. "${phrase}"`);
            });
        }
        
        if (Object.keys(data.domainExpertise || {}).length > 0) {
            console.log('\n🧠 Domain Expertise:');
            Object.entries(data.domainExpertise).forEach(([domain, level]) => {
                console.log(`   • ${domain}: ${level}`);
            });
        }
        
        if (data.communicationPatterns) {
            console.log('\n📊 Communication Patterns:');
            console.log(`   Question Style: ${data.communicationPatterns.questionStyle || 'direct'}`);
            console.log(`   Response Length: ${data.communicationPatterns.responseLength || 'medium'}`);
        }
        
        console.log('\n' + '═'.repeat(60) + '\n');
        
    } catch (error) {
        console.error(`❌ Error: ${error.response?.data?.error || error.message}`);
        process.exit(1);
    }
}

async function exportTrainingData() {
    try {
        const response = await axios.get(`${API_URL}/api/training/export`, {
            responseType: 'stream'
        });
        
        const filename = `training-data-${Date.now()}.json`;
        const filepath = path.join(process.cwd(), filename);
        const writer = fs.createWriteStream(filepath);
        
        response.data.pipe(writer);
        
        return new Promise((resolve, reject) => {
            writer.on('finish', () => {
                console.log(`✅ Exported training data to: ${filename}`);
                console.log(`   Use this file for fine-tuning your models`);
                resolve();
            });
            writer.on('error', reject);
        });
    } catch (error) {
        console.error(`❌ Error: ${error.response?.data?.error || error.message}`);
        process.exit(1);
    }
}

// Interactive mode
async function interactiveMode() {
    const readline = require('readline').createInterface({
        input: process.stdin,
        output: process.stdout
    });
    
    const question = (prompt) => new Promise((resolve) => {
        readline.question(prompt, resolve);
    });
    
    console.log('\n🎓 Personal Training Data Manager - Interactive Mode\n');
    
    while (true) {
        console.log('\nOptions:');
        console.log('  1. Add training data');
        console.log('  2. List training data');
        console.log('  3. Remove training data');
        console.log('  4. Export training data');
        console.log('  5. Exit\n');
        
        const choice = await question('Choose an option (1-5): ');
        
        if (choice === '1') {
            console.log('\nAvailable types:');
            Object.entries(TYPES).forEach(([key, value]) => {
                console.log(`  • ${key}: ${value}`);
            });
            const type = await question('\nType: ');
            const data = await question('Data: ');
            await addTrainingData(type, data);
        } else if (choice === '2') {
            await listTrainingData();
        } else if (choice === '3') {
            const type = await question('Type to remove: ');
            const data = await question('Data to remove: ');
            await removeTrainingData(type, data);
        } else if (choice === '4') {
            await exportTrainingData();
        } else if (choice === '5') {
            console.log('\n👋 Goodbye!');
            readline.close();
            break;
        }
    }
}

// CLI mode
async function main() {
    const args = process.argv.slice(2);
    const command = args[0];
    
    if (!command || command === 'interactive' || command === 'i') {
        await interactiveMode();
        return;
    }
    
    switch (command) {
        case 'add':
            if (args.length < 3) {
                console.error('Usage: node train-me.js add <type> <data>');
                console.error('\nTypes:', Object.keys(TYPES).join(', '));
                process.exit(1);
            }
            await addTrainingData(args[1], args.slice(2).join(' '));
            break;
            
        case 'remove':
            if (args.length < 3) {
                console.error('Usage: node train-me.js remove <type> <data>');
                process.exit(1);
            }
            await removeTrainingData(args[1], args.slice(2).join(' '));
            break;
            
        case 'list':
            await listTrainingData();
            break;
            
        case 'export':
            await exportTrainingData();
            break;
            
        default:
            console.log(`
Personal Training Data Manager

Usage:
  node train-me.js <command> [args]

Commands:
  add <type> <data>     Add training data
  remove <type> <data>  Remove training data
  list                  List all training data
  export                Export training data for fine-tuning
  interactive           Interactive mode

Types:
  ${Object.entries(TYPES).map(([k, v]) => `  ${k.padEnd(20)} ${v}`).join('\n  ')}

Examples:
  node train-me.js add writing_style "I prefer concise explanations"
  node train-me.js add personal_fact "I'm a software engineer"
  node train-me.js list
  node train-me.js export
            `);
    }
}

if (require.main === module) {
    main().catch(console.error);
}

module.exports = { addTrainingData, removeTrainingData, listTrainingData, exportTrainingData };

