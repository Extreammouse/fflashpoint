#!/bin/bash
# Run full Flash Point pipeline
# Usage: bash bin/run_pipeline.sh

set -e

echo "🚀 Flash Point Pipeline Launcher"
echo "=================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found${NC}"
    exit 1
fi

if ! command -v java &> /dev/null; then
    echo -e "${RED}❌ Java not found${NC}"
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo -e "${RED}❌ Virtual environment not found. Run: python3 -m venv .venv${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites OK${NC}"

# Activate venv
echo -e "${YELLOW}Activating virtual environment...${NC}"
source .venv/bin/activate

# Check Kafka
echo -e "${YELLOW}Checking Kafka...${NC}"
if ! docker ps | grep -q kafka; then
    echo -e "${YELLOW}Starting Kafka...${NC}"
    docker-compose -f docker/docker-compose.yml up -d
    sleep 5
    echo -e "${GREEN}✅ Kafka started${NC}"
else
    echo -e "${GREEN}✅ Kafka already running${NC}"
fi

# Create topics
echo -e "${YELLOW}Creating Kafka topics...${NC}"
CONTAINER=$(docker ps --filter "name=kafka" --format "{{.Names}}" | head -1)
if [ -z "$CONTAINER" ]; then
    echo -e "${RED}❌ Kafka container not found${NC}"
    exit 1
fi

docker exec "$CONTAINER" kafka-topics --create --topic gaze_events \
    --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1 \
    --if-not-exists 2>/dev/null || true

docker exec "$CONTAINER" kafka-topics --create --topic intervention_trigger \
    --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1 \
    --if-not-exists 2>/dev/null || true

echo -e "${GREEN}✅ Kafka topics ready${NC}"

# Instructions
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Infrastructure ready!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Start components in separate terminals:"
echo ""
echo -e "${YELLOW}Terminal 2: Spark Processor${NC}"
echo "  source .venv/bin/activate && \\"
echo "  export JAVA_HOME=\$(/usr/libexec/java_home -v 17) && \\"
echo "  /opt/homebrew/opt/apache-spark/bin/spark-submit \\"
echo "    --py-files src/processors/gaze_udf.py \\"
echo "    --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1 \\"
echo "    src/processors/spark_analytics.py"
echo ""
echo -e "${YELLOW}Terminal 3: Vision Producer${NC}"
echo "  source .venv/bin/activate && python src/producers/vision_node.py"
echo ""
echo -e "${YELLOW}Terminal 4: Flash Consumer${NC}"
echo "  source .venv/bin/activate && python src/consumers/flash_trigger.py"
echo ""
echo -e "${YELLOW}Terminal 5: Stub Producer (for testing)${NC}"
echo "  source .venv/bin/activate && python src/producers/stub_producer.py"
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
