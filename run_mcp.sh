#!/bin/zsh
cd /Users/cashify/Projects/multi-modal-rag-project
set -a
source .env
set +a
exec .venv/bin/python server.py
