# gunicorn.conf.py
import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
backlog = 2048

# Worker processes
workers = 1  # 使用單個 worker 來避免 SocketIO 問題
worker_class = 'eventlet'  # 使用 eventlet worker 類別來支持 WebSocket
worker_connections = 1000
max_requests = 0  # 禁用 worker 重啟
max_requests_jitter = 0

# Timeout settings - 重要：增加超時設置
timeout = 300  # 5分鐘超時
keepalive = 30
graceful_timeout = 300
worker_timeout = 300  # 與 timeout 保持一致

# Memory management
preload_app = True  # 預加載應用程序
max_requests_per_child = 0  # 禁用基於請求數的重啟

# Logging
loglevel = 'warning'  # 減少日志輸出
access_log_format = '%(h)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" in %(D)sµs'
accesslog = '-'
errorlog = '-'
capture_output = True

# Process naming
proc_name = 'flask_socketio_app'

# Security
limit_request_line = 8190
limit_request_fields = 100
limit_request_field_size = 8190

# SSL (如果需要)
# keyfile = None
# certfile = None

# 重要：添加這些設置來處理 SocketIO
def when_ready(server):
    server.log.info("Server is ready. Spawning workers")

def worker_int(worker):
    worker.log.info("worker received INT or QUIT signal")

def pre_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_worker_init(worker):
    worker.log.info("Worker initialized (pid: %s)", worker.pid)

def worker_abort(worker):
    worker.log.info("Worker received SIGABRT signal")

# 環境變量
raw_env = [
    'PYTHONPATH=/opt/render/project/src',
]

# 安全設置
user = None
group = None
tmp_upload_dir = None

# 性能調優
forwarded_allow_ips = '*'
proxy_protocol = False
proxy_allow_ips = '*'

# 其他設置
pythonpath = '/opt/render/project/src'
chdir = '/opt/render/project/src'

# 針對 Render 平台的優化
if os.environ.get('RENDER'):
    # Render 平台特定設置
    bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
    workers = 1  # Render 建議使用單個 worker
    timeout = 0  # Render 平台上禁用超時
    worker_timeout = 0
