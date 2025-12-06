#!/bin/bash

echo "Starting Docker services..."

# Start services in detached mode
docker-compose up -d

echo ""
echo "✓ Services started!"
echo ""
echo "🌐 API:        http://localhost:8000"
echo "📊 API Docs:   http://localhost:8000/docs"
echo "🔍 Health:     http://localhost:8000/health"
echo ""
echo "View logs:"
echo "  API:         docker-compose logs -f fraud-api"
echo "  Monitoring:  docker-compose logs -f monitoring"
echo ""
echo "Stop services: docker-compose down"