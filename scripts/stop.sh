#!/bin/bash
echo "Running stop.sh..." >> /home/ubuntu/stop.log
pkill -f app.py || echo "No app.py process found" >> /home/ubuntu/stop.log
echo "stop.sh completed." >> /home/ubuntu/stop.log
exit 0
