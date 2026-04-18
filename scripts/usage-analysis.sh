#!/bin/bash
# scripts/usage-analysis.sh
# Performs Day 6 Capacity Planning & Growth analysis

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

DATABASE_URL=${DATABASE_URL:-"postgresql://ollama:password@localhost:5432/ollama"}

echo "📊 OLLAMA ELITE - CAPACITY PLANNING & USAGE ANALYSIS (Jan 18, 2026)"
echo "------------------------------------------------------------------"

echo "1. API Request Volume (Last 24 Hours)"
psql $DATABASE_URL -c "
  SELECT DATE_TRUNC('hour', created_at) as hour,
         COUNT(*) as requests,
         AVG(response_time_ms)::INT as avg_latency_ms
  FROM usage
  WHERE created_at > NOW() - INTERVAL '24 hours'
  GROUP BY hour
  ORDER BY hour DESC;
"

echo ""
echo "2. Top Models by Usage"
psql $DATABASE_URL -c "
  SELECT model, COUNT(*) as requests,
         SUM(input_tokens) as total_input_tokens,
         SUM(output_tokens) as total_output_tokens
  FROM usage
  WHERE created_at > NOW() - INTERVAL '24 hours'
  GROUP BY model
  ORDER BY requests DESC;
"

echo ""
echo "3. Error Rate Analysis"
psql $DATABASE_URL -c "
  SELECT status_code, COUNT(*) as occurrences
  FROM usage
  WHERE created_at > NOW() - INTERVAL '24 hours'
  GROUP BY status_code
  ORDER BY occurrences DESC;
"

echo ""
echo "4. Cache Efficiency (Metrics based)"
# Assuming we can query the metrics endpoint or have them in DB
# For now, let's look at usage metadata if we store cache hits there
psql $DATABASE_URL -c "
  SELECT usage_metadata->>'cache_hit_type' as cache_type, COUNT(*) as count
  FROM usage
  WHERE created_at > NOW() - INTERVAL '24 hours'
  AND usage_metadata->>'cache_hit' = 'true'
  GROUP BY cache_type;
"

echo "------------------------------------------------------------------"
echo "✅ Analysis Complete. Proceeding to Growth Forecasting."
