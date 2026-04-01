#!/bin/bash
# Forex AI Agent — service control script
# Usage: ./scripts/service.sh {start|stop|status|logs|restart}

LABEL="com.forex.agent"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG_DIR="$HOME/projects/forex-ai-agent/logs"
LOG_FILE="$LOG_DIR/system.log"

mkdir -p "$LOG_DIR"

case "$1" in
    start)
        # Kill any manually-started instance on port 8000
        kill $(lsof -t -i :8000) 2>/dev/null
        sleep 1

        # Bootstrap prices into Redis before starting
        cd "$HOME/projects/forex-ai-agent"
        /Users/testinggeeks/.pyenv/versions/3.11.9/bin/python3 -m scripts.bootstrap_prices 2>&1 | tail -1

        launchctl load "$PLIST" 2>/dev/null
        launchctl start "$LABEL"
        sleep 2

        if launchctl list | grep -q "$LABEL"; then
            echo "Forex AI Agent started"
            echo "Dashboard: http://localhost:8000/api/v1/status"
            echo "Logs: tail -f $LOG_FILE"
        else
            echo "Failed to start — check: launchctl list | grep forex"
        fi
        ;;

    stop)
        launchctl stop "$LABEL" 2>/dev/null
        launchctl unload "$PLIST" 2>/dev/null
        # Clean up any orphaned process
        kill $(lsof -t -i :8000) 2>/dev/null
        echo "Forex AI Agent stopped"
        ;;

    restart)
        "$0" stop
        sleep 3
        "$0" start
        ;;

    status)
        if launchctl list | grep -q "$LABEL"; then
            PID=$(launchctl list | grep "$LABEL" | awk '{print $1}')
            echo "Forex AI Agent: RUNNING (PID: $PID)"

            # Check dashboard health
            HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null)
            if [ "$HEALTH" = "200" ]; then
                echo "Dashboard: healthy (http://localhost:8000)"
            else
                echo "Dashboard: not responding (HTTP $HEALTH)"
            fi

            # Show price feed status
            curl -s http://localhost:8000/api/v1/status 2>/dev/null | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    pf=d.get('price_feed',{})
    print(f\"Price feed: {pf.get('provider','?')} — {pf.get('tick_count',0)} ticks\")
    agents=d.get('agents',{})
    for name,info in agents.items():
        ms=info.get('last_execution_ms','?')
        print(f\"  {name}: {ms}ms\")
except: print('  (could not parse status)')
" 2>/dev/null
        else
            echo "Forex AI Agent: STOPPED"
        fi
        ;;

    logs)
        if [ -f "$LOG_FILE" ]; then
            tail -f "$LOG_FILE"
        else
            echo "No log file at $LOG_FILE"
        fi
        ;;

    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
