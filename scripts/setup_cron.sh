#!/bin/bash
# Setup script for Nostrus cron jobs
# Run this script to install the scheduled automation jobs

NOSTRUS_HOME="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_PATH="${NOSTRUS_HOME}/venv/bin/python"

echo "Setting up Nostrus automation..."
echo "Project directory: ${NOSTRUS_HOME}"

# Check if virtual environment exists
if [ ! -f "${PYTHON_PATH}" ]; then
    echo "Error: Virtual environment not found at ${NOSTRUS_HOME}/venv"
    echo "Please create it first: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Create cron job entries
CRON_JOBS="
# Nostrus Football Betting Model Automation
# =========================================

# Data Collection: Monday and Thursday at 06:00
0 6 * * 1 cd ${NOSTRUS_HOME} && ${PYTHON_PATH} -m src.automation.collect_data >> logs/cron.log 2>&1
0 6 * * 4 cd ${NOSTRUS_HOME} && ${PYTHON_PATH} -m src.automation.collect_data >> logs/cron.log 2>&1

# Model Retraining: First Monday of each month at 07:00
0 7 1-7 * 1 [ \$(date +\\%d) -le 7 ] && cd ${NOSTRUS_HOME} && ${PYTHON_PATH} -m src.automation.train_model >> logs/cron.log 2>&1

# Prediction Generation: Friday and Tuesday at 08:00
0 8 * * 5 cd ${NOSTRUS_HOME} && ${PYTHON_PATH} -m src.automation.generate_predictions >> logs/cron.log 2>&1
0 8 * * 2 cd ${NOSTRUS_HOME} && ${PYTHON_PATH} -m src.automation.generate_predictions >> logs/cron.log 2>&1

# Weekly log rotation (Sunday midnight)
0 0 * * 0 find ${NOSTRUS_HOME}/logs -name \"*.log\" -size +10M -exec gzip {} \\;
"

echo ""
echo "The following cron jobs will be added:"
echo "${CRON_JOBS}"
echo ""
read -p "Do you want to install these cron jobs? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Backup existing crontab
    crontab -l > /tmp/crontab_backup_$(date +%Y%m%d_%H%M%S) 2>/dev/null || true

    # Add new jobs
    (crontab -l 2>/dev/null | grep -v "Nostrus"; echo "${CRON_JOBS}") | crontab -

    echo "Cron jobs installed successfully!"
    echo ""
    echo "Current crontab:"
    crontab -l
else
    echo "Installation cancelled."
fi
