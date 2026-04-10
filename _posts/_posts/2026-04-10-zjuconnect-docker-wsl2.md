---
title: Zjuconnect_Docker_WSL2
date: 2026-04-10 11:30:00 +0800
categories: [tools]
tags: [network]     # TAG names should always be lowercase
math: true
mermaid: true
---

# 🐛 Debug 记录：zju-connect (aTrust) WSL2+Docker Desktop 验证码访问死锁与持久化部署

## 1. 环境信息
*   **软件版本**：mythologyli/zju-connect:latest
*   **协议**：aTrust
*   **运行环境**：Windows 宿主机 + WSL2（Ubuntu 发行版） + Docker Desktop
*   **部署方式**：Docker Compose

## 2. 初始目标
参考配置部署 zju-connect 客户端，对接 aTrust 协议 VPN，实现容器内登录凭据持久化，完成一次性交互式登录后，可长期后台无感运行代理服务。

## 3. 问题排查过程

### ❌ 踩坑 1：本地镜像版本非最新，存在兼容性风险
*   **现象**：执行 `docker run` 启动容器后，通过 `docker images` 发现本地镜像为 2 个月前的旧版本，与 Docker Hub 远程 latest 标签的镜像摘要不一致。
*   **原因**：Docker 不会自动拉取 `latest` 标签的最新镜像，仅会使用本地已存在的同名镜像，导致本地版本落后。
*   **修复**：手动执行 `docker pull mythologyli/zju-connect:latest` 强制拉取远程最新镜像，并对比本地与远程镜像摘要确认版本一致。

### ❌ 踩坑 2：无头环境下验证码服务无法访问
*   **现象**：执行 `docker compose run --rm -it zju-connect` 触发交互式登录，日志报错 `Failed to open browser: exec: "xdg-open": executable file not found in $PATH. Please visit: http://127.0.0.1:36833`；WSL2 内执行 `curl http://127.0.0.1:36833` 报错 `Connection refused`，Windows 宿主机浏览器也无法打开该地址。
*   **原因**：
    1.  容器内无图形界面工具 `xdg-open`，无法自动弹出浏览器。
    2.  查阅源码发现，zju-connect 验证码服务启动时强制执行 `net.Listen("tcp", "127.0.0.1:0")`，强制绑定容器本地回环地址（仅容器内部可访问），且端口随机分配，常规 `-p` 端口映射仅能转发监听在 `0.0.0.0` 的端口，对该地址完全无效。

### ❌ 踩坑 3：`network_mode: host` 配置不生效
*   **现象**：在 `docker-compose.yml` 中添加 `network_mode: "host"` 后重新启动容器，WSL2 内执行 `ss -tuln | grep 36833` 无任何输出，端口未监听；通过 `docker inspect` 查看容器实际网络模式，发现为 `bridge` 而非配置的 `host`。
*   **原因**：
    1.  Docker 官方设计行为：`docker compose run` 启动的临时容器，**不会继承 service 中定义的 network_mode 网络配置**，必须通过 `--network=host` 参数显式指定。
    2.  Docker Desktop 网络隔离限制：Windows/WSL2 环境下，`network_mode: host` 绑定的是 Docker 背后隐藏的 Linux 虚拟机的本地回环地址，而非 WSL2 虚拟机/Windows 宿主机的回环地址，导致端口无法跨网络访问。

### ❌ 踩坑 4：常规端口映射方案完全失效
*   **现象**：尝试通过 `-p` 提前映射端口，但由于验证码端口随机分配，无法在 `docker-compose.yml` 中写死规则；即使临时映射，也因服务绑定 `127.0.0.1` 而无法访问。
*   **原因**：验证码服务强制绑定 `127.0.0.1` 且端口随机，常规端口映射无法突破容器本地回环限制。

---

## 4. 最终破局方案

**核心思路**：“一次性原生方式完成交互式登录，生成持久化凭据后，交由 Docker 长期稳定后台运行”，既规避 Docker 网络隔离的核心卡点，又保证长期运行的稳定性。

**具体步骤**：
1. **更新镜像至最新版本**
   * 停止并清理旧容器：`docker stop zju-connect && docker rm zju-connect`
   * 强制拉取远程最新镜像：`docker pull mythologyli/zju-connect:latest`
   * 验证版本一致性：对比本地与远程镜像摘要，确认已更新至最新版。

2. **配置 Docker Compose 持久化环境**
   编写 `docker-compose.yml`，核心配置本地目录挂载，保证登录凭据可持久化、跨容器复用：
   ```yaml
   version: '3.8'

   services:
     zju-connect:
       container_name: zju-connect
       image: mythologyli/zju-connect
       restart: unless-stopped
       ports:
         - "1080:1080"  # socks5 代理端口
         - "1081:1081"  # 管理端口
       configs:
         - source: zju-connect-config
           target: /home/nonroot/config.toml
       volumes:
         # 核心：本地目录挂载，Windows/WSL2 均可直接访问，凭据持久化
         - ./data:/home/nonroot/data

   configs:
     zju-connect-config:
       content: |
         username = ""
         password = ""
         client_data_file = "/home/nonroot/data/client_data.json"
         protocol = "atrust"
         # 验证码图片本地保存路径，兜底方案使用
         graph_code_file = "/home/nonroot/data/graph_code.jpg"
   ```

3. **一次性交互式登录（2种可选方案，均适配WSL2环境）**
   #### 方案A：nsenter 网络转发（零侵入首选，无需改配置）
   * 提前创建本地数据目录：`mkdir -p ./data`
   * 启动交互容器，**显式指定host网络**，保持终端全程打开，直到日志输出验证码地址：
     ```bash
     docker compose run --rm -it --network=host zju-connect
     ```
   * 新开 WSL2 终端，执行一键转发命令，将容器内 127.0.0.1 的随机验证码端口转发到 WSL2 固定端口：
     ```bash
     # 自动获取容器PID与验证码端口
     CONTAINER_PID=$(docker inspect -f '{{.State.Pid}}' $(docker ps -q --filter "name=zju-connect"))
     CAPTCHA_PORT=$(docker logs $(docker ps -q --filter "name=zju-connect") | grep -oP '127.0.0.1:\K\d+' | tail -1)
     echo "✅ 验证码端口: $CAPTCHA_PORT"

     # 安装转发工具（仅首次执行）
     sudo apt update && sudo apt install -y socat

     # 启动端口转发
     sudo nsenter -t $CONTAINER_PID -n socat TCP-LISTEN:18080,fork,reuseaddr TCP:127.0.0.1:$CAPTCHA_PORT
     ```
   * Windows 宿主机浏览器访问 `http://<WSL2_IP>:18080`（WSL2_IP 可通过 `ip addr show eth0` 获取），完成图形验证码验证。
   * 回到交互容器终端，输入短信验证码，看到 `Client data saved to /home/nonroot/data/client_data.json` 提示后，按 `Ctrl+C` 退出容器与转发进程。

   #### 方案B：本地图片坐标验证（兜底方案，完全规避网络问题）
   * 提前创建本地数据目录：`mkdir -p ./data`
   * 直接启动交互容器：`docker compose run --rm -it zju-connect`
   * 终端提示输入验证码时，打开 Windows 资源管理器，访问路径 `\\wsl$\<你的WSL发行版名>\home\<用户名>\zjuvpn\data`，找到 `graph_code.jpg` 验证码图片。
   * 用 Windows 经典画图工具打开图片，按 `Ctrl+1` 一键切换到 100% 缩放比例，按图片提示顺序获取文字中心的像素坐标。
   * 在终端按格式输入坐标 JSON（无空格），例如 `[[181,66],[250,80]]`，回车后按提示输入短信验证码，完成登录。

4. **Docker 长期后台运行**
   * 确认 `./data` 目录下已生成 `client_data.json` 登录凭据文件。
   * 后台启动容器，无需再配置 host 网络，直接使用端口映射：
     ```bash
     docker compose up -d
     ```
   * 验证服务状态：
     - 查看容器运行状态：`docker ps`
     - 查看服务日志：`docker logs zju-connect`，确认 `VPN client started` 无报错
     - 代理连通性测试：`curl -x socks5://127.0.0.1:1080 https://example.com`
