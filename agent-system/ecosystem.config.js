module.exports = {
    apps: [{
        name: 'agent-terminal',
        script: './server.js',
        instances: 1,
        autorestart: true,
        watch: false,
        max_memory_restart: '500M',
        error_file: './logs/pm2-error.log',
        out_file: './logs/pm2-out.log',
        log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
        env: {
            NODE_ENV: 'development',
            PORT: 3000,
            LOG_LEVEL: 'info'
        },
        env_production: {
            NODE_ENV: 'production',
            PORT: 3000,
            LOG_LEVEL: 'warn',
            SESSION_SECRET: 'CHANGE_ME_IN_PRODUCTION'
        }
    }]
};
