#!/bin/bash
# Production Training Monitor
# Usage: ./monitor_training.sh

echo "═══════════════════════════════════════════════════════════════"
echo "🚀 CELESTIAL PRODUCTION TRAINING MONITOR"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Find training process
TRAIN_PID=$(ps aux | grep train_celestial_production | grep -v grep | awk '{print $2}' | head -1)

if [ -z "$TRAIN_PID" ]; then
    echo "❌ Training process NOT RUNNING!"
    echo ""
    echo "Last log entries:"
    tail -20 logs/production_training_*.log 2>/dev/null || echo "No logs found"
    exit 1
fi

echo "✅ Training process RUNNING (PID: $TRAIN_PID)"
echo ""

# Show process stats
echo "📊 PROCESS STATISTICS:"
ps aux | grep $TRAIN_PID | grep -v grep | awk '{printf "   CPU: %s%%   MEM: %s%% (%s KB)   Runtime: %s\n", $3, $4, $6, $10}'
echo ""

# Show memory stats
echo "🧠 SYSTEM MEMORY:"
free -h | grep -E "Mem:|Swap:" | awk '{printf "   %-6s Total: %-8s Used: %-8s Free: %-8s\n", $1, $2, $3, $4}'
echo ""

# Find latest log file
LOG_FILE=$(ls -t logs/production_training_*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "❌ No log file found!"
    exit 1
fi

echo "📝 TRAINING PROGRESS (from $LOG_FILE):"
echo ""

# Extract latest batch info
LATEST_BATCH=$(grep "BATCH" "$LOG_FILE" | tail -1)
if [ -n "$LATEST_BATCH" ]; then
    echo "   $LATEST_BATCH"
    
    # Calculate progress
    CURRENT_BATCH=$(echo "$LATEST_BATCH" | grep -oP 'BATCH \K[0-9]+' | head -1)
    TOTAL_BATCHES=$(echo "$LATEST_BATCH" | grep -oP 'BATCH [0-9]+/\K[0-9]+' | head -1)
    CURRENT_EPOCH=$(echo "$LATEST_BATCH" | grep -oP 'Epoch \K[0-9]+' | head -1)
    TOTAL_EPOCHS=$(echo "$LATEST_BATCH" | grep -oP 'Epoch [0-9]+/\K[0-9]+' | head -1)
    
    if [ -n "$CURRENT_BATCH" ] && [ -n "$TOTAL_BATCHES" ]; then
        BATCH_PROGRESS=$(awk "BEGIN {printf \"%.2f\", ($CURRENT_BATCH/$TOTAL_BATCHES)*100}")
        echo "   Progress: $BATCH_PROGRESS% of Epoch $CURRENT_EPOCH/$TOTAL_EPOCHS"
    fi
else
    echo "   No batch information found yet (still initializing?)"
fi

echo ""
# Smoothed loss (last 200 and last 50)
LAST200_LOSS=$(grep "BATCH" "$LOG_FILE" | awk -F'|' '{print $3}' | grep -oP 'loss=\K[0-9.]+' | tail -200)
LAST50_LOSS=$(echo "$LAST200_LOSS" | tail -50)
SMOOTH200=$(echo "$LAST200_LOSS" | awk '{sum+=$1; n++} END { if(n>0) printf "%.4f", sum/n; else print "N/A" }')
SMOOTH50=$(echo "$LAST50_LOSS" | awk '{sum+=$1; n++} END { if(n>0) printf "%.4f", sum/n; else print "N/A" }')

# Smoothed dir_acc (last 200 and last 50)
LAST200_ACC=$(grep "BATCH" "$LOG_FILE" | awk -F'|' '{print $4}' | grep -oP 'dir_acc=\K[0-9.]+(?=%)' | tail -200)
LAST50_ACC=$(echo "$LAST200_ACC" | tail -50)
ACC200=$(echo "$LAST200_ACC" | awk '{sum+=$1; n++} END { if(n>0) printf "%.2f%%", sum/n; else print "N/A" }')
ACC50=$(echo "$LAST50_ACC" | awk '{sum+=$1; n++} END { if(n>0) printf "%.2f%%", sum/n; else print "N/A" }')

echo "📈 SMOOTHED LOSS:  last200=$SMOOTH200   last50=$SMOOTH50"
echo "🎯 SMOOTHED DIR_ACC: last200=$ACC200   last50=$ACC50"

echo ""
echo "📈 RECENT LOSS TREND (last 10 batches):"
grep "BATCH" "$LOG_FILE" | tail -10 | awk -F'|' '{print "   " $3}' | grep -oP 'loss=\K[0-9.]+'

echo ""
echo "🎯 RECENT DIRECTIONAL ACCURACY (last 10 batches):"
grep "BATCH" "$LOG_FILE" | tail -10 | awk -F'|' '{print "   " $4}' | grep -oP 'dir_acc=\K[0-9.]+%'

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "💡 Commands:"
echo "   Watch logs:      tail -f $LOG_FILE"
echo "   Full log:        cat $LOG_FILE"
echo "   Stop training:   kill $TRAIN_PID"
echo "   Memory check:    free -h && ps aux | grep $TRAIN_PID"
echo "═══════════════════════════════════════════════════════════════"
