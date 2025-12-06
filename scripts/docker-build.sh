#!/bin/bash

echo "Building Docker images..."

# Build the image
docker-compose build

echo "✓ Build complete!"
echo ""
echo "Run 'docker-compose up' to start services"