#!/usr/bin/env bash
set -euo pipefail

# 1. OS 패키지 업데이트
sudo apt-get update

# 2. Docker 설치
sudo apt-get install -y docker.io

# 3. Docker 서비스 시작
sudo service docker start

# 4. Docker 동작 확인
docker version
docker info

# 5. Python 패키지 설치
python -m pip install --upgrade pip
python -m pip install harbor

# 6. Harbor 실행
harbor run -d terminal-bench/terminal-bench-2 -a oracle