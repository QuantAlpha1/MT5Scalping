# MT5 Scalper Bot - Installation Guide

## Table of Contents
1. [System Requirements](#system-requirements)
2. [MetaTrader 5 Setup](#metatrader-5-setup)
3. [Python Environment Setup](#python-environment-setup)
4. [Dependencies Installation](#dependencies-installation)
5. [Configuration](#configuration)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum
- Windows 10/11 or Linux (Wine compatible)
- MetaTrader 5 build 2250+
- Python 3.8+
- 4GB RAM
- 2GB free disk space

### Recommended
- Windows 10/11 Pro
- MetaTrader 5 build 2560+
- Python 3.10+
- 8GB RAM
- SSD storage

---

## MetaTrader 5 Setup

1. **Install MT5 Terminal**
   - Download from your broker's website
   - Complete standard installation

2. **Enable API Access**
   - Open MT5 → Tools → Options → Expert Advisors
   - Enable:
     - ☑ Allow automated trading
     - ☑ Allow DLL imports
     - ☑ Allow WebRequest for listed URL

3. **Add Symbols**
   - Right-click Market Watch → Show All
   - Add at least:
     - EURUSD
     - USDJPY

4. **Verify Terminal Info**
   - Press F9 to open Terminal window
   - Confirm:
     - Build number ≥ 2250
     - Connection status: "Connected"

---

## Python Environment Setup

### Windows
```bash
# 1. Install Python
winget install Python.Python.3.10

# 2. Create virtual environment
python -m venv .venv

# 3. Activate environment
.\.venv\Scripts\activate
